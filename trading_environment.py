import numpy as np
import pandas as pd
import gymnasium as gym
import joblib
from collections import OrderedDict
from preprocess_data import Indicator
from typing import Optional

def get_safe_rl_inputs(model, returns):
    """
    Calculates the probability of being in each state for every time step
    using ONLY past data (Forward Algorithm), preventing look-ahead bias.
    """
    n_states = model.n_components
    belief_history = []
    
    log_curr_belief = np.log(model.startprob_)
    belief_history.append(model.startprob_)
    
    for ret in returns:
        log_emission = np.zeros(n_states)
        for s in range(n_states): 
            mean = model.means_[s, 0]
            var = model.covars_[s, 0]
            term = -0.5 * np.log(2 * np.pi * var)
            diff = (ret - mean)**2
            log_emission[s] = term - (diff / (2 * var))
            
        curr_belief = np.exp(log_curr_belief)
        predicted_state = curr_belief @ model.transmat_
        new_belief_unweighted = predicted_state * np.exp(log_emission)
        
        sum_belief = np.sum(new_belief_unweighted)
        if sum_belief < 1e-12:
             new_belief = np.ones(n_states) / n_states
        else:
             new_belief = new_belief_unweighted / sum_belief
        
        belief_history.append(new_belief)
        log_curr_belief = np.log(new_belief + 1e-10)

    return np.array(belief_history[:len(returns)])


class TradingEnv(gym.Env):
    def __init__(self, data, timestamp=0, hidden_markov=None, tree_model=None):
        super().__init__()
        self._timestamp = timestamp

        cols = [
            "vol_ratio", "realized_vol", "realized_vol_60", "ADX", "bandwidth",
            "KAMA_res", "MFI", "Dir_Ind_Diff", "relative_volume", "ADOSC",
            "smoothed_BOP", "RSI_fast", "usd_score_wstd", "5m_forward_returns",
        ]
        feature_cols = cols[:-1]
        
        data[feature_cols] = data[feature_cols].fillna(0.0)
        self.data = data.loc[:, cols].copy()
        
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data = self.data.ffill().bfill()

        self._xs = self.data.loc[:, feature_cols].copy()
        self._feature_arrays = {col: self.data[col].values for col in self._xs.columns}

        self._h = hidden_markov
        self._m = tree_model

        safe_returns = self.data["5m_forward_returns"].shift(5).fillna(0.0).to_numpy()
        self._regimes = get_safe_rl_inputs(self._h, safe_returns)
        self._regimes = np.nan_to_num(self._regimes, nan=0.33, posinf=1.0, neginf=0.0)

        print("Pre-computing XGBoost predictions...")
        self._xgb_predictions = self._m.predict(self._xs.values).astype(np.float32)
        self._xgb_predictions = np.nan_to_num(self._xgb_predictions, 
                                              nan=0.0, posinf=0.0, neginf=0.0)
        print(f"Cached {len(self._xgb_predictions)} predictions")

        self._cache = OrderedDict()
        self._max_cache_entries = 50000
        self._ind = Indicator(data, self._timestamp)
        self._price_cache = np.exp(self._ind.close).astype(np.float32)

        # ============== WEALTH & POSITION PARAMETERS ==============
        self.starting_wealth = 1e6
        self.current_wealth = self.starting_wealth
        
        self._peak_wealth = self.starting_wealth
        self._max_drawdown_pct = 0.0
        
        self._free_capital = self.starting_wealth
        self.max_episode_length = 1440
        
        self._dd_guard = 0.2
        
        self._transaction_cost_pct = 1e-5
        self._accumulated_costs = 0.0
        
        self._max_leverage = 1.0
        self._current_position = 0.0
        
        self._ret_clip = 0.1
        
        # ============== EMA SHARPE PARAMETERS ==============
        # EMA decay factor (alpha) for Sharpe calculation
        # 
        # For 1-minute EURUSD bars:
        # - Key volatility cycles: 1-4 hours (session-based)
        # - Episode length: 1440 min (24 hours)
        # - Want memory spanning ~1 session transition
        #
        # Half-life = ln(2) / alpha
        # alpha = 0.0115 → half-life ≈ 60 minutes (1 hour)
        # alpha = 0.0058 → half-life ≈ 120 minutes (2 hours)
        #
        # Using 90-minute half-life as compromise:
        # alpha = ln(2) / 90 ≈ 0.0077
        self._ema_alpha = 0.0077
        
        # EMA of returns (first moment)
        self._ema_mean = 0.0
        
        # EMA of squared returns (second moment)
        self._ema_var = 1e-8
        
        # Previous EMA Sharpe for delta calculation
        self._prev_ema_sharpe = 0.0
        
        # Current EMA Sharpe
        self._ema_sharpe = 0.0
        
        # Minimum variance floor
        self._var_floor = 1e-10
        
        # Scale factor for Sharpe delta rewards
        self._sharpe_reward_scale = 200.0
        
        # Hard clamp on step reward
        self._reward_clip = 5.0
        
        # Warm-up steps (should be ~1-2 half-lives for EMA stability)
        # 90 min half-life → warm-up of ~100-180 steps
        self._warmup_steps = 120
        
        # ============== TRACKING VARIABLES ==============
        self._active_steps = 0
        self._total_steps = 0
        
        self._sharpe = 0.0  # Cumulative Sharpe (for reporting)
        self._sharpe_annualized = 0.0  # For reporting only
        self._losses = 0.0
        
        self._PnLs = []  # List of step returns (for cumulative Sharpe reporting)
        
        self._terminated_due_to_drawdown = False
        
        # Annualization factor - ONLY for reporting
        self._annualization_factor = np.sqrt(252 * 24 * 60)
        
        # ============== OBSERVATION SPACE ==============
        def _finite_min(a, default=-1.0):
            a = np.asarray(a, dtype=float)
            m = np.nanmin(a)
            return default if not np.isfinite(m) else float(m)

        def _finite_max(a, default=1.0):
            a = np.asarray(a, dtype=float)
            m = np.nanmax(a)
            return default if not np.isfinite(m) else float(m)

        low_list = [(_finite_min(self.data[col].to_numpy()) - 1.0) 
                    for col in self._xs.columns]
        low_list += [_finite_min(self._ind.close, default=0.0), -10.0, -10.0, 0.0, 0.0, 0.0]
        low_limits = np.array(low_list, dtype=float)

        high_list = [(_finite_max(self.data[col].to_numpy()) + 1.0) 
                     for col in self._xs.columns]
        high_list += [_finite_max(self._ind.close, default=1.0), 10.0, 10.0, 1.0, 1.0, 1.0]
        high_limits = np.array(high_list, dtype=float)

        self.observation_space = gym.spaces.Dict(
            {
                "indicators" : gym.spaces.Box(low=np.array(low_limits, dtype=np.float32), 
                                              high=np.array(high_limits, dtype=np.float32), 
                                              shape=(19,)),
                "agent" : gym.spaces.Box(low=np.array([-10.0, -1.0, 0.0, -1.0], dtype=np.float32), 
                                         high=np.array([10.0, 1.0, 1.0, 1.0], dtype=np.float32), 
                                         shape=(4,))
            }
        )

        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
    
    def set_max_episode_length(self, length):
        self.max_episode_length = length
    
    def _predict_model(self, time):
        """Return cached XGBoost prediction."""
        return float(self._xgb_predictions[time])

    def _get_current_leverage(self):
        """Calculate current leverage as position_value / wealth."""
        current_price = self._price_cache[self._timestamp]
        position_value = self._current_position * current_price
        return position_value / self.current_wealth

    def _update_ema_sharpe(self, new_return: float) -> float:
        """
        Update EMA mean/variance and compute new EMA Sharpe.
        
        Uses Welford-style online update for numerical stability.
        
        Args:
            new_return: The latest step return
            
        Returns:
            Current EMA Sharpe ratio
        """
        alpha = self._ema_alpha
        
        # Update EMA of returns (first moment)
        self._ema_mean = alpha * new_return + (1 - alpha) * self._ema_mean
        
        # Update EMA of variance (second moment)
        # Using: Var = E[X²] - E[X]²
        # But for EMA, we track variance directly for stability
        deviation_sq = (new_return - self._ema_mean) ** 2
        self._ema_var = alpha * deviation_sq + (1 - alpha) * self._ema_var
        
        # Compute EMA Sharpe
        std = np.sqrt(max(self._ema_var, self._var_floor))
        self._ema_sharpe = self._ema_mean / std
        
        return self._ema_sharpe

    def _compute_cumulative_sharpe(self) -> float:
        """
        Compute cumulative Sharpe for reporting purposes.
        
        Returns:
            Cumulative Sharpe ratio, or 0.0 if insufficient data
        """
        if len(self._PnLs) < 20:
            return 0.0
        
        returns = np.array(self._PnLs)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        if std_ret < 1e-10:
            return 0.0
        
        return float(mean_ret / std_ret)

    def _get_obs(self):
        """Convert internal state to observation format."""
        indicator_list = [float(self._feature_arrays[col][self._timestamp]) 
                          for col in self._xs.columns]
        
        prev_timestamp = max(0, self._timestamp - 1)
        indicator_list.extend([
            float(self._ind.close[self._timestamp]), 
            float(self._predict_model(prev_timestamp)),
            float(self._predict_model(self._timestamp)),
            float(self._regimes[self._timestamp][0]), 
            float(self._regimes[self._timestamp][1]), 
            float(self._regimes[self._timestamp][2])
        ])
    
        indicators = np.array(indicator_list, dtype=np.float32)
        indicators = np.nan_to_num(indicators, nan=0.0, posinf=1e6, neginf=-1e6)
    
        log_return = np.log(self.current_wealth / self.starting_wealth)
        
        current_drawdown_pct = ((self._peak_wealth - self.current_wealth) 
                                / (self._peak_wealth + 1e-12))
        current_drawdown_pct = max(0.0, min(1.0, current_drawdown_pct))
        
        current_leverage = self._get_current_leverage()
        
        # Use EMA Sharpe for observation (bounded by tanh)
        # EMA Sharpe typical range: -1 to +1, so scale by 2
        bounded_sharpe = float(np.tanh(self._ema_sharpe * 2.0))
        
        agent_status = np.array([
            float(log_return), 
            bounded_sharpe, 
            float(current_drawdown_pct),
            float(np.clip(current_leverage, -1.0, 1.0))
        ], dtype=np.float32)
        
        return {"indicators": indicators, "agent": agent_status}
    
    def _get_info(self):
        """Compute auxiliary information for debugging."""
        current_leverage = self._get_current_leverage()
        
        # Compute cumulative Sharpe for reporting
        cumulative_sharpe = self._compute_cumulative_sharpe()
        
        return {
           "current_wealth" : self.current_wealth,
           "free_capital" : self._free_capital,
           "accumulated_costs" : self._accumulated_costs,
           "losses" : self._losses,
           "sharpe" : cumulative_sharpe,  # Cumulative for reporting
           "ema_sharpe" : self._ema_sharpe,  # EMA Sharpe (used for reward)
           "sharpe_annualized" : cumulative_sharpe * self._annualization_factor,
           "peak_wealth" : self._peak_wealth,
           "max_drawdown_pct" : self._max_drawdown_pct,
           "active_steps" : self._active_steps,
           "total_steps" : self._total_steps,
           "activity_rate" : self._active_steps / max(self._total_steps, 1),
           "current_leverage" : current_leverage,
           "current_position" : self._current_position,
           "step_return" : self._PnLs[-1] if self._PnLs else 0.0,
           "ema_mean" : self._ema_mean,
           "ema_var" : self._ema_var,
        }

    def _compute_ema_sharpe_delta_reward(self, new_return: float) -> float:
        """
        Compute reward as the CHANGE in EMA Sharpe ratio.
        
        Using EMA Sharpe ensures:
        1. Constant responsiveness throughout episode
        2. Recent performance weighted more heavily
        3. No signal shrinkage as episode progresses
        
        Args:
            new_return: The latest step return
            
        Returns:
            Scaled EMA Sharpe delta, clipped to prevent outliers
        """
        # Store previous Sharpe
        prev_sharpe = self._prev_ema_sharpe
        
        # Update EMA and get new Sharpe
        current_sharpe = self._update_ema_sharpe(new_return)
        
        # Update previous for next step
        self._prev_ema_sharpe = current_sharpe
        
        # During warm-up, return small reward based on return sign
        # This prevents unstable Sharpe deltas before EMA stabilizes
        if self._total_steps < self._warmup_steps:
            return float(np.clip(new_return * 10.0, -self._reward_clip, self._reward_clip))
        
        # Compute delta
        sharpe_delta = current_sharpe - prev_sharpe
        
        # Scale the delta
        scaled_reward = sharpe_delta * self._sharpe_reward_scale
        
        # Clip to prevent outliers
        clipped_reward = float(np.clip(scaled_reward, -self._reward_clip, self._reward_clip))
        
        return clipped_reward

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode."""
        super().reset(seed=seed)

        max_start = len(self.data) - self.max_episode_length - 1
        
        if max_start > 0:
            self._timestamp = self.np_random.integers(0, max_start)
        else:
            self._timestamp = 0
            
        self._ind.timestamp = self._timestamp
        
        self.current_wealth = self.starting_wealth
        self._free_capital = self.starting_wealth
        
        self._peak_wealth = self.starting_wealth
        self._max_drawdown_pct = 0.0
        self._terminated_due_to_drawdown = False
        
        self._active_steps = 0
        self._total_steps = 0
        self._accumulated_costs = 0.0
        
        self._current_position = 0.0
        
        # Reset EMA Sharpe tracking
        self._ema_mean = 0.0
        self._ema_var = 1e-8  # Small positive for stability
        self._ema_sharpe = 0.0
        self._prev_ema_sharpe = 0.0
        
        # Reset reporting variables
        self._sharpe = 0.0
        self._sharpe_annualized = 0.0
        
        self._losses = 0.0
        self._cache.clear()
        
        self._PnLs = []

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _compute_terminal_reward(self):
        """Compute terminal reward/penalty based on EMA Sharpe."""
        terminal_reward = 0.0
        
        # Penalty for drawdown breach
        if self._terminated_due_to_drawdown:
            terminal_reward -= 5.0
        
        # Bonus/penalty based on final EMA Sharpe
        # Good EMA Sharpe: > 0.1 (consistent profitability)
        # Bad EMA Sharpe: < 0 (losing money)
        if self._ema_sharpe > 0.1:
            terminal_reward += 2.0
        elif self._ema_sharpe < 0.0:
            terminal_reward -= 2.0
            
        return terminal_reward

    def step(self, action):
        """Execute one timestep with TARGET LEVERAGE control and EMA Sharpe Delta reward."""
        
        # 1. Parse target leverage
        target_leverage = float(np.clip(action[0], -self._max_leverage, self._max_leverage))
        
        current_price = self._price_cache[self._timestamp]
        
        # 2. Calculate target position (in integer units)
        target_position_float = (target_leverage * self.current_wealth) / current_price
        target_position = int(np.round(target_position_float))
        
        # Calculate integer trade size
        trade_size = target_position - int(np.round(self._current_position))
        
        # 3. Calculate costs
        trade_cost = abs(trade_size) * current_price * self._transaction_cost_pct
        
        # 4. Update position
        self._current_position = float(target_position)
        self._accumulated_costs += trade_cost
        
        # Recalculate actual leverage after rounding
        actual_leverage = (self._current_position * current_price) / self.current_wealth
        leverage_at_t = abs(actual_leverage)
        
        # Track activity
        if leverage_at_t > 0.01:
            self._active_steps += 1
        
        # 5. Advance time
        t_next = self._timestamp + 1
        
        if t_next >= len(self._price_cache):
            truncated = True
            terminated = False
            reward = self._compute_terminal_reward()
            return self._get_obs(), reward, terminated, truncated, self._get_info()
            
        price_next = self._price_cache[t_next]
        
        # 6. Compute step return (including transaction costs)
        price_return = (price_next - current_price) / current_price
        step_ret = actual_leverage * price_return
        
        # Subtract transaction cost as percentage of wealth
        cost_drag = trade_cost / self.current_wealth
        step_ret -= cost_drag
        
        # Clip extreme returns
        step_ret = float(np.clip(step_ret, -self._ret_clip, self._ret_clip))
        
        # 7. Update tracking
        self._total_steps += 1
        self._PnLs.append(step_ret)
        
        # 8. Compute EMA Sharpe Delta reward
        reward = self._compute_ema_sharpe_delta_reward(step_ret)
        
        # 9. Advance state
        self._timestamp = t_next
        self._ind.timestamp = self._timestamp
        
        # Update wealth using sum of returns
        cumulative_return = sum(self._PnLs)
        self.current_wealth = self.starting_wealth * (1.0 + cumulative_return)
        self._peak_wealth = max(self._peak_wealth, self.current_wealth)
        
        current_drawdown_pct = (self._peak_wealth - self.current_wealth) / (self._peak_wealth + 1e-12)
        current_drawdown_pct = max(0.0, min(1.0, current_drawdown_pct))
        
        self._max_drawdown_pct = max(self._max_drawdown_pct, current_drawdown_pct)
        self._losses = self._peak_wealth - self.current_wealth
        
        position_value = abs(self._current_position) * price_next
        self._free_capital = max(self.current_wealth - position_value, 0)
            
        # 10. Termination checks
        terminated = self.current_wealth >= 1e9
        drawdown_breach = current_drawdown_pct >= self._dd_guard
        wealth_floor_breach = self.current_wealth <= 0.8 * self.starting_wealth
        length_limit = self._total_steps >= self.max_episode_length
        
        if drawdown_breach:
            self._terminated_due_to_drawdown = True
            
        truncated = length_limit or drawdown_breach or wealth_floor_breach
        
        # 11. Final reward
        if terminated or truncated:
            reward = reward + self._compute_terminal_reward()
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
