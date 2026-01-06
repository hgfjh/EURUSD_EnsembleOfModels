import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm


ITERABLE_FEATURE_SET = ["RSI", "MFI", "STOCH", "ADX", "AROON", "ADOSC", 
                        "cos_time_cycle", "sin_time_cycle", "realized_vol", 
                        "relative_volume"]

# This class contains the entire indicator suite as methods
class Indicator:

    def __init__(self, data, timestamp):
        self.data = data
        self.timestamp = timestamp
        self._cache = OrderedDict()
        self._max_cache_entries = 50000

        self.close = list(np.log(self.data["bid_close"]))
        self.high = list(np.log(self.data["bid_high"]))
        self.low  = list(np.log(self.data["bid_low"]))
        self.open = list(np.log(self.data["bid_open"]))
        self.volume = list(self.data["volume"])

        # Robust intraday phase based on actual UTC timestamps
        # Parse once; store minute-of-day as int array
        if "minute_utc" in self.data.columns:
            dt = pd.to_datetime(self.data["minute_utc"], utc=True, errors="coerce")
            self.minute_of_day = (dt.dt.hour * 60 + dt.dt.minute).astype("Int64")
        else:
            self.minute_of_day = None

    def _cache_get(self, key):
        val = self._cache.get(key)
        if val is not None:
            # mark as recently used
            try:
                self._cache.move_to_end(key)
            except Exception:
                pass
        return val
    
    def _cache_set(self, key, val):
        # insert / update and mark as recently used
        self._cache[key] = val
        try:
            self._cache.move_to_end(key)
        except Exception:
            pass
        # evict oldest if over limit
        if len(self._cache) > self._max_cache_entries:
            self._cache.popitem(last=False)
        return val

    # Simple Moving Average indicator
    def SMA(self, window, series = None):
        # if series provided, average last "window" entries of it
        if series is None:
            if self.timestamp < (window - 1):
                return np.nan
            prices = self.close[self.timestamp - window + 1:self.timestamp + 1]
            return sum(prices) / window
        else:
            s = list(series)
            if len(s) < window:
                return np.nan
            return sum(s[-window:]) / window
    
    # Exponential Moving Average Indicator (Incremental)
    def EMA(self, window, time = None, series = None):
        smoothing = 2.0 / (window + 1)
        
        # Fallback for custom series (non-incremental unless handled by caller)
        if series is not None:
            data = list(series)
            if len(data) == 0: return np.nan
            if time is None: time = len(data) - 1
            ema = data[0]
            for i in range(1, time + 1):
                ema = smoothing * data[i] + (1 - smoothing) * ema
            return ema
            
        # Incremental logic for self.close
        if time is None: time = self.timestamp
        if time < 0: return np.nan
        
        key = ("EMA", window, time)
        cached = self._cache_get(key)
        if cached is not None: return cached
        
        if time == 0:
            return self._cache_set(key, self.close[0])
            
        # Recursive step: O(1) if sequential
        prev_ema = self.EMA(window, time - 1)
        val = smoothing * self.close[time] + (1 - smoothing) * prev_ema
        return self._cache_set(key, val)
    
    # Weighted Moving Average Indicator
    def WMA(self, window):
        # Return NaN if timestamp < window - 1; otherwise, compute WMA
        if self.timestamp < (window - 1):
            return np.nan
        result = 0.0
        for i in range(window):
            result += ((window - i) * self.close[self.timestamp - i])
        result = result / ((window * (window + 1)) / 2)
        return result
    
    # Efficiency Ratio
    def ER(self, window):
        if self.timestamp < (window - 1):
            return np.nan
        prices = self.close[self.timestamp - window + 1:self.timestamp + 1]
        total_move = 0.0
        for i in range(1, len(prices)):
            total_move += abs(prices[i] - prices[i-1])
        if total_move == 0:
             return 0.0
        return abs(prices[-1] - prices[0]) / total_move

    # Kaufman's Adaptive Moving Average
    def KAMA(self, window = 10, time = None, slow = 30, fast = 2):
        if time is None:
            time = self.timestamp
        if time < (window - 1):
            return np.nan
        key = ("KAMA", window, slow, fast, time)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        
        # Base case
        if time == window - 1:
            initial_kama = sum(self.close[0:window]) / window
            return self._cache_set(key, initial_kama)

        # Recursive step: Get previous KAMA
        prev_kama = self.KAMA(window, time - 1, slow, fast)
        
        # Calculate ER for current step only (O(window))
        # We need prices from time-window to time
        prices = self.close[time - window : time + 1]
        change = abs(prices[-1] - prices[0])
        volatility = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices)))
        er = change / volatility if volatility != 0 else 0.0
        
        sc_base = er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)
        sc = sc_base * sc_base
        
        current_kama = prev_kama + sc * (self.close[time] - prev_kama)
        return self._cache_set(key, current_kama)
    
    # Rolling Volume Weighted Average Price
    def VWAP(self, window = 30, time = None):
        if time is None:
            time = self.timestamp
        if time < (window - 1):
            return np.nan
        key = ("VWAP", window, time)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        prices = self.close[time - window + 1:time + 1]
        volumes = self.volume[time - window + 1:time + 1]
        numerator = 0.0
        denominator = 0.0
        for price, volume in zip(prices, volumes):
            numerator += (price * volume)
            denominator += volume
        result = (numerator / denominator) if denominator != 0 else np.nan
        return self._cache_set(key, result)
        
    #Moving Average Convergence/Divergence
    def MACD(self, slow = 20, fast = 5, time = None, series = None):
        return self.EMA(fast, time, series) - self.EMA(slow, time, series)
    
    # Helper for incremental MACD Signal
    def _MACD_Signal(self, slow, fast, signal, time):
        key = ("MACD_Signal", slow, fast, signal, time)
        cached = self._cache_get(key)
        if cached is not None: return cached
        
        # Calculate MACD line (difference of EMAs)
        macd_t = self.EMA(fast, time) - self.EMA(slow, time)
        
        if time == 0:
            return self._cache_set(key, macd_t)
            
        # Signal is EMA of MACD line
        prev_sig = self._MACD_Signal(slow, fast, signal, time - 1)
        alpha = 2.0 / (signal + 1)
        sig = alpha * macd_t + (1 - alpha) * prev_sig
        return self._cache_set(key, sig)

    # Signal and histogram for MACD (Incremental)
    def signal_and_histogram(self, slow = 18, fast = 6, signal = 6):
        t = self.timestamp
        s = self._MACD_Signal(slow, fast, signal, t)
        # Recompute MACD line for current time to get histogram
        m = self.EMA(fast, t) - self.EMA(slow, t)
        return [s, m - s]
    
    #Fast %K
    def STOCHF_k(self, window = 14, time = None):
        if time is None:
            time = self.timestamp
        if time < (window - 1):
            return np.nan
            
        key = ("STOCHF_k", window, time)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
            
        prices = self.close[time - window + 1:time + 1]
        low = min(prices)
        high = max(prices)
        denom = (high - low)
        if denom == 0.0:
            k = 0.0
        else:
            k = 100.0 * ((self.close[time] - low) / denom)
        return self._cache_set(key, k)
    
    #Fast %D
    def STOCHF(self, window = 14, time = None, smooth = 3):
        if time is None:
            time = self.timestamp
        key = ("STOCHF", window, smooth, time)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        series = [self.STOCHF_k(window, time - i) for i in range(smooth)]
        result = self.SMA(smooth, series)
        return self._cache_set(key, result)
    
    #Slow %D
    def STOCH(self, window = 14, smooth = 3):
        key = ("STOCH", window, smooth, self.timestamp)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        series = [self.STOCHF(window, self.timestamp - i, smooth) for i in range(smooth)]
        result = self.SMA(smooth, series)
        return self._cache_set(key, result)
    
    #Aroon Oscillator
    def AROON(self, window = 25, time = None):
        if time is None:
            time = self.timestamp
        # Return NaN if time < window - 1; otherwise, return AROON
        if time < (window - 1):
            return np.nan
        key = ("AROON", window, time)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        prices = self.close[time - window + 1:time + 1]
        recent_high = max(prices)
        recent_low = min(prices)

        days_since_high = None
        days_since_low = None
        for idx in range(len(prices) - 1, -1, -1):
            if prices[idx] == recent_high and days_since_high is None:
                days_since_high = len(prices) - 1 - idx
            if prices[idx] == recent_low and days_since_low is None:
                days_since_low = len(prices) - 1 - idx
            if days_since_high is not None and days_since_low is not None:
                break

        days_since_high = days_since_high if days_since_high is not None else 0
        days_since_low = days_since_low if days_since_low is not None else 0

        a_up = 100.0 * ((window - days_since_high) / window)
        a_down = 100.0 * ((window - days_since_low) / window)

        result = a_up - a_down
        return self._cache_set(key, result)
    
    #Bollinger Bands (Windowed)
    def BB(self, window = 20, k = 2):
        t = self.timestamp
        if t < (window - 1):
            return np.nan
        # Optimization: Slice only the needed window
        slice_data = self.close[t - window + 1 : t + 1]
        series = pd.Series(slice_data)
        
        mb = series.mean()
        std = series.std(ddof=0)
        
        if std == 0 or np.isnan(std):
            return [0.0, 0.5, 0.0]
        upper = mb + k * std
        lower = mb - k * std
        if (upper - lower) == 0:
            return [0.0, 0.5, 0.0]
        stretch = (self.close[t] - mb) / std
        percent_B = (self.close[t] - lower) / (upper - lower)
        bandwidth = 2 * k * std
        return [float(stretch), float(percent_B), float(bandwidth)]
    
    #Chaikin's Accumulation/Distribution Line (Incremental)
    def AD(self, time = None):
        if time is None:
            time = self.timestamp
        key = ("AD", time)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
            
        if time == 0:
            return self._cache_set(key, 0.0)
            
        # Incremental step: AD[t] = AD[t-1] + MFM * Volume
        prev_ad = self.AD(time - 1)
        
        h = self.high[time]
        l = self.low[time]
        c = self.close[time]
        v = self.volume[time]
        
        denom = h - l
        if denom == 0:
            mfm = 0.0
        else:
            mfm = (2 * c - h - l) / denom
            
        flow = mfm * v
        return self._cache_set(key, prev_ad + flow)
    
    # Helper for incremental EMA of AD line
    def _EMA_of_AD(self, span, time):
        key = ("EMA_AD", span, time)
        cached = self._cache_get(key)
        if cached is not None: return cached
        
        ad_val = self.AD(time)
        
        if time == 0:
            return self._cache_set(key, ad_val)
            
        prev = self._EMA_of_AD(span, time - 1)
        alpha = 2.0 / (span + 1)
        val = alpha * ad_val + (1 - alpha) * prev
        return self._cache_set(key, val)

    #Chaikin’s Accumulation/Distribution Oscillator (Incremental)
    def ADOSC(self, slow = 10, fast = 3, time = None):
        if time is None:
            time = self.timestamp
        if time < 30:
            return np.nan
            
        # ADOSC is simply EMA(fast) of AD - EMA(slow) of AD
        fast_ema = self._EMA_of_AD(fast, time)
        slow_ema = self._EMA_of_AD(slow, time)
        return float(fast_ema - slow_ema)
    
    # Helper for incremental RSI components
    def _rsi_smooth(self, window, time, is_gain):
        key = ("RSI_smooth", window, time, is_gain)
        cached = self._cache_get(key)
        if cached is not None: return cached
        
        if time == 0:
            return self._cache_set(key, 0.0)
            
        diff = self.close[time] - self.close[time - 1]
        val = diff if is_gain else -diff
        val = max(0.0, val)
        
        prev = self._rsi_smooth(window, time - 1, is_gain)
        alpha = 1.0 / window
        # Wilder's smoothing: (prev * (n-1) + curr) / n  
        # =>  prev * (1 - alpha) + curr * alpha
        res = prev * (1 - alpha) + val * alpha
        return self._cache_set(key, res)

    #Relative Strength Index (Incremental)
    def RSI(self, window = 14, time = None):
        if time is None:
            time = self.timestamp
        if time < (window - 1):
            return np.nan
        
        avg_gain = self._rsi_smooth(window, time, True)
        avg_loss = self._rsi_smooth(window, time, False)
        
        if avg_loss == 0:
            if avg_gain == 0: return 50.0
            return 100.0
            
        rs = avg_gain / avg_loss
        return float(100.0 * rs / (1.0 + rs))
    
    #True Range
    def TR(self, time = None):
        # compute TR for a single timestamp using vectorized slice
        if time is None:
            time = self.timestamp
        key = ("TR", time)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        if time == 0:
            return self._cache_set(key, float(self.high[0] - self.low[0]))
        h = self.high[time]
        l = self.low[time]
        pc = self.close[time - 1]
        result = float(max(h - l, abs(h - pc), abs(l - pc)))
        return self._cache_set(key, result)
    
    #Average True Range
    def ATR(self, window = 14, time = None):
        # Optimization: ATR is exactly Wilder's smoothed TR. 
        # Use the incremental smoothed_TR method instead of rebuilding full series.
        return self.smoothed_TR(window, time)
    
    #Normalized Average True Range
    def NATR(self, window = 14):
        if self.timestamp < (window - 1):
            return np.nan
        # Since ATR is calculated on Log-Prices, it is already normalized relative to price.
        # Just scale it to percentage terms.
        return self.ATR(window) * 100.0
    
    #Positive Directional Movement
    def POS_DM(self, time = None):
        if time is None:
            time = self.timestamp
        if time == 0:
            return 0.0
        up_move = (self.high[time] - self.high[time - 1])
        down_move = (self.low[time - 1] - self.low[time])
        if up_move > down_move and up_move > 0:
             return up_move
        else:
            return 0.0
    
    #Negative Directional Movement
    def NEG_DM(self, time = None):
        if time is None:
            time = self.timestamp
        if time == 0:
            return 0.0
        up_move = (self.high[time] - self.high[time - 1])
        down_move = (self.low[time - 1] - self.low[time])
        if down_move > up_move and down_move > 0:
            return down_move
        else:
            return 0.0

    #"Wilder-smoothed" True Range
    def smoothed_TR(self, window = 14, time = None):
        if time is None:
            time = self.timestamp
        key = ("smoothed_TR", window, time)
        cached = self._cache_get(key)
        if time == 0:
            return self._cache_set(key, self.TR(0))
        if cached is not None:
            return cached
        span = 2 * window - 1
        alpha = 2.0 / (span + 1)
        prev = self._cache_get(("smoothed_TR", window, time - 1))
        if prev is not None:
            tr_now = self.TR(time)
            val = prev * (1 - alpha) + alpha * tr_now
            return self._cache_set(key, val)
        # Cache the full TR series to avoid rebuilding the list
        key_series = ("TR_series", time)
        tr_series_cached = self._cache_get(key_series)
        if tr_series_cached is None:
            trs = [self.TR(i) for i in range(0, time + 1)]
            tr_series_cached = trs
            self._cache_set(key_series, tr_series_cached)
        else:
            trs = tr_series_cached
        val = self.EMA(span, time = len(trs) - 1, series = trs)
        return self._cache_set(key, val)
    
    #"Wilder-smoothed" Positive Directional Movement
    def smoothed_POS_DM(self, window = 14, time = None):
        if time is None:
            time = self.timestamp
        key = ("smoothed_POS_DM", window, time)
        cached = self._cache_get(key)
        if time == 0:
            return self._cache_set(key, 0.0)
        if cached is not None:
            return cached
        span = 2 * window - 1
        alpha = 2.0 / (span + 1)
        prev = self._cache_get(("smoothed_POS_DM", window, time - 1))
        if prev is not None:
            move = self.POS_DM(time)
            val = prev * (1 - alpha) + alpha * move
            return self._cache_set(key, val)
        key_series = ("POS_DM_series", time)
        moves_cached = self._cache_get(key_series)
        if moves_cached is None:
            moves = [self.POS_DM(i) for i in range(0, time + 1)]
            moves_cached = moves
            self._cache_set(key_series, moves_cached)
        else:
            moves = moves_cached
        val = self.EMA(span, time = len(moves) - 1, series = moves)
        return self._cache_set(key, val)

    #"Wilder-smoothed" Negative Directional Movement
    def smoothed_NEG_DM(self, window = 14, time = None):
        if time is None:
            time = self.timestamp
        key = ("smoothed_NEG_DM", window, time)
        cached = self._cache_get(key)
        if time == 0:
            return self._cache_set(key, 0.0)
        if cached is not None:
            return cached
        span = 2 * window - 1
        alpha = 2.0 / (span + 1)
        prev = self._cache_get(("smoothed_NEG_DM", window, time - 1))
        if prev is not None:
            move = self.NEG_DM(time)
            val = prev * (1 - alpha) + alpha * move
            return self._cache_set(key, val)
        key_series = ("NEG_DM_series", time)
        moves_cached = self._cache_get(key_series)
        if moves_cached is None:
            moves = [self.NEG_DM(i) for i in range(0, time + 1)]
            moves_cached = moves
            self._cache_set(key_series, moves_cached)
        else:
            moves = moves_cached
        val = self.EMA(span, time = len(moves) - 1, series = moves)
        return self._cache_set(key, val)

    #Positive Directional Indicator
    def POS_DI(self, window = 14, time = None):
        if time is None:
            time = self.timestamp
        key = ("POS_DI", window, time)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        denom = self.smoothed_TR(window, time)
        if denom == 0 or np.isnan(denom):
            return self._cache_set(key, 0.0)
        result = 100.0 * (self.smoothed_POS_DM(window, time) / denom)
        return self._cache_set(key, result)
    
    #Negative Directional Indicator
    def NEG_DI(self, window = 14, time = None):
        if time is None:
            time = self.timestamp
        key = ("NEG_DI", window, time)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        denom = self.smoothed_TR(window, time)
        if denom == 0 or np.isnan(denom):
            return self._cache_set(key, 0.0)
        result = 100.0 * (self.smoothed_NEG_DM(window, time) / denom)
        return self._cache_set(key, result)
    
    #Directional Index
    def DX(self, window = 14, time = None):
        if time is None:
            time = self.timestamp
        key = ("DX", window, time)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        pos = self.POS_DI(window, time)
        neg = self.NEG_DI(window, time)
        if (pos + neg) == 0:
            return self._cache_set(key, 0.0)
        result = (100.0 * abs(pos - neg) / (pos + neg))
        return self._cache_set(key, result)
    
    #Wilder-smoothed DX (Incremental)
    def ADX(self, window = 14, time = None):
        if time is None:
            time = self.timestamp
        if time < (2 * window - 1):
            return np.nan
        key = ("ADX", window, time)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
            
        # Base case: First ADX value
        if time == 2 * window - 1:
            dxs = [self.DX(window, i) for i in range(window, time + 1)]
            initial_adx = sum(dxs) / window
            return self._cache_set(key, initial_adx)
            
        # Incremental step: ADX[t] = (ADX[t-1] * (w-1) + DX[t]) / w
        # We trust that ADX(time-1) is already cached or easily computed
        prev_adx = self.ADX(window, time - 1)
        current_dx = self.DX(window, time)
        adx = (prev_adx * (window - 1) + current_dx) / window
        return self._cache_set(key, adx)
    
    #Momentum
    def MOM(self, window = 5, time = None):
        if time is None:
            time = self.timestamp
        if time < window:
            return np.nan
        key = ("MOM", window, time)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        mom = (self.close[time] 
                - self.close[time - window])
        return self._cache_set(key, mom)
    
    #Balance of Power
    def BOP(self, time = None, window = 10, smoothed = True):
        if not smoothed:
            if time is None:
                time = self.timestamp
            high = self.high[time]
            low = self.low[time]
            if (high - low) == 0:
                return 0.0
            return ((self.close[time] - self.open[time]) / (high - low))
        else:
            if time is None:
                time = self.timestamp
            if time < (window - 1):
                return np.nan
            key = ("BOP_smoothed", window, time)
            cached = self._cache_get(key)
            if cached is not None:
                return cached
            start = time - window + 1
            key_series = ("BOP_series", window, time)
            balances = self._cache_get(key_series)
            if balances is None:
                balances = [self.BOP(i, smoothed=False) for i in range(start, time + 1)]
                self._cache_set(key_series, balances)
            result = self.EMA(window, time=len(balances) - 1, series=balances)
            return self._cache_set(key, result)
    
    def delta_BOP(self, window = 10, smoothed = True):
        return (self.BOP(self.timestamp, window, smoothed) 
                - self.BOP(self.timestamp - 1, window, smoothed))       
    
    #Typical Price
    def typical_price(self, time = None):
        if time is None:
            time = self.timestamp
        return ((self.high[time] + self.low[time] + self.close[time]) / 3)
    
    #Mean Deviation of smoothed price from price 
    #used to compute Commodity Channel Index
    #We use tanh to keep outputs bounded even in extreme situations
    def CCI(self, window = 20, k = 200):
        if self.timestamp < (window - 1):
            return np.nan
        t = self.timestamp
        key = ("CCI", window, k, t)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        key_tp = ("typical_price_series", window, t)
        tps = self._cache_get(key_tp)
        if tps is None:
            tps = [self.typical_price(i) for i in range(t - window + 1, t + 1)]
            self._cache_set(key_tp, tps)
        smoothed_price = self.SMA(window, series=tps)
        deviations = [abs(tp - smoothed_price) for tp in tps]
        mean_abs_deviation = self.SMA(window, series=deviations)
        if mean_abs_deviation == 0:
            return self._cache_set(key, 0.0)
        result = np.tanh(((self.typical_price() - smoothed_price) 
                         / (0.015 * mean_abs_deviation)) / k)
        return self._cache_set(key, result)

    #Raw Money Flow    
    def RMF(self, time = None):
        if time is None:
            time = self.timestamp   
        return (self.close[time] * self.volume[time])

    #Money Flow Index (Windowed)
    def MFI(self, window = 14):
        t = self.timestamp
        if t < window: 
            return np.nan
            
        start = t - window
        # We need window+1 points to get 'window' diffs
        highs = self.high[start : t+1]
        lows = self.low[start : t+1]
        closes = self.close[start : t+1]
        vols = self.volume[start : t+1]
        
        tp = (pd.Series(highs) + pd.Series(lows) + pd.Series(closes)) / 3.0
    
        rmf = np.exp(tp) * pd.Series(vols)
        
        tp_delta = tp.diff()
        
        # tp_delta has window+1 elements, first is NaN. 
        pos_flow = rmf.where(tp_delta > 0.0, 0.0).iloc[1:].sum()
        neg_flow = rmf.where(tp_delta < 0.0, 0.0).iloc[1:].sum()
        
        if neg_flow == 0.0:
            if pos_flow == 0.0: return 50.0
            return 100.0
        mf_ratio = pos_flow / neg_flow
        return float(100.0 - (100.0 / (1.0 + mf_ratio)))
    
    def forward_return(self, window):
        if self.timestamp < (len(self.close) - window):
            return self.close[self.timestamp + window] - self.close[self.timestamp]
        else:
            return np.nan
    
    def cos_time_cycle(self):
        """
        Intraday cycle based on clock time (UTC), not row index.
        Returns cos(2π * minute_of_day / 1440). Falls back to row index if needed.
        """
        if self.minute_of_day is None:
            return float(np.cos((np.pi / 720.0) * self.timestamp))

        m = self.minute_of_day.iat[self.timestamp]
        if pd.isna(m):
            return float(np.cos((np.pi / 720.0) * self.timestamp))

        return float(np.cos(np.pi * (int(m) / 720.0)))

    def sin_time_cycle(self):
        if self.minute_of_day is None:
            return float(np.sin((np.pi / 720.0) * self.timestamp))

        m = self.minute_of_day.iat[self.timestamp]
        if pd.isna(m):
            return float(np.sin((np.pi / 720.0) * self.timestamp))

        return float(np.sin(np.pi * (int(m) / 720.0)))

    def realized_vol(self, window = 20, time = None):
        if time is None:
            time = self.timestamp
        if time < window:
            return np.nan

        key = ("realized_vol", window, time)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        # window+1 closes -> window one-step log returns
        closes = np.asarray(self.close[time - window : time + 1], dtype=float)
        rets = np.diff(closes)  # length == window
        val = float(np.sqrt(np.dot(rets, rets)))
        return self._cache_set(key, val)
    
    def relative_volume(self, window = 20):
        if self.timestamp < (window - 1):
            return np.nan
        volumes = self.volume[self.timestamp - window + 1:self.timestamp + 1]
        avg_vol = self.SMA(window, volumes)
        if avg_vol  == 0:
            return np.nan
        return (self.volume[self.timestamp] / avg_vol)

    
def main():
    df = pd.read_csv("EURUSD_price_plus_ravenpack_1min_merged_with_shared_channel.csv")

    m = pd.to_datetime(df["minute_utc"], utc=True, errors="coerce")
    dow = m.dt.weekday  # Mon=0 ... Sun=6

    # one-hot
    dow_oh = pd.get_dummies(dow, prefix="dow", dtype="int8")


    dow_oh = dow_oh.drop(columns=["dow_5"], errors="ignore")

    df = pd.concat([df, dow_oh], axis=1)

    #Number of contiguous 1 minute blocks
    gap = (m.diff() != pd.Timedelta(minutes=1)) | m.isna()
    seg_id = gap.cumsum()
    df["segment_id"] = seg_id

    #Number of Lags
    LAG = 3

    # Precompute how many rows we will process across all segments (for tqdm total)
    LOOKBACK = 60
    HORIZON = 5

    segments = []
    total_rows = 0
    for _, df_seg in df.groupby(seg_id, sort=False):
        df_seg = df_seg.reset_index().rename(columns={"index": "orig_idx"})
        n = len(df_seg)
        if n <= LOOKBACK + HORIZON:
            continue
        start = LOOKBACK
        end = n - HORIZON 
        total_rows += (end - start)
        segments.append(df_seg)

    pbar = tqdm(total=total_rows, desc="Building DF", unit="row")

    inds = {"RSI": [], "RSI_fast" : [], "RSI_slow" : [], "MFI": [], "STOCH": [], 
            "ADX": [], "AROON": [], "ADOSC": [],"stretch": [], "%B": [], 
            "bandwidth": [], "stretch_5": [], "%B_5": [],"bandwidth_5": [], 
            "MACD_hist": [], "Dir_Ind_Diff": [], "BOP": [], "delta_BOP" : [], 
            "smoothed_BOP": [], "delta_sm_BOP" : [], "MOM_1": [], "MOM_3": [], 
            "MOM_5": [], "MOM_10": [], "VWAP_res": [], "KAMA_res": [], 
            "cos_time_cycle" : [], "sin_time_cycle" : [], "realized_vol" : [], 
            "realized_vol_60":[], "relative_volume" : [], "bid_range" : [], 
            "ADOSC_lag_1" : [], "ADOSC_lag_2" : [], "ADOSC_lag_3" : [],
            "AROON_lag_1" : [], "AROON_lag_2" : [], "AROON_lag_3" : [],
            "RSI_lag_1" : [], "RSI_lag_2" : [], "RSI_lag_3" : [],
            "vol_ratio" : [], "delta_VWAP_res" : [], "delta_KAMA_res" : [],
            "5m_forward_returns" : []}
    
    rows_kept = []

    # Collect outputs across segments
    for df_seg in segments:
        orig_idx = df_seg["orig_idx"].to_numpy()

        df_seg_ind = df_seg.drop(columns=["orig_idx"]).reset_index(drop=True)
        ind = Indicator(df_seg_ind, 0)

        # compute min_vol on this segment
        sigmas = [ind.realized_vol(time=i) for i in range(len(ind.close))]
        min_vol = 0.1 * pd.Series(sigmas).median()
        sigmas_60 = [ind.realized_vol(60, i) for i in range(len(ind.close))]
        min_vol_60 = 0.1 * pd.Series(sigmas_60).median()

        # only iterate where both lookback and horizon are valid inside the segment
        for i in range(LOOKBACK, len(df_seg_ind) - HORIZON):
            rows_kept.append(orig_idx[i])   

            ind.timestamp = i
            sigma = ind.realized_vol()
            sigma_60 = ind.realized_vol(60)
            rv = np.nan if np.isnan(sigma) else max(sigma, min_vol)
            rv_60 = np.nan if np.isnan(sigma_60) else max(sigma_60, min_vol_60) 

            for func_name in ITERABLE_FEATURE_SET:
                func = getattr(ind, func_name)
                inds[func_name].append(func())
            bb = ind.BB()
            if isinstance(bb, (list, tuple, np.ndarray)):
                inds["stretch"].append(bb[0])
                inds["%B"].append(bb[1])
                inds["bandwidth"].append(bb[2])
            else:
                inds["stretch"].append(np.nan)
                inds["%B"].append(np.nan)
                inds["bandwidth"].append(np.nan)
            bb_5 = ind.BB(5)
            if isinstance(bb_5, (list, tuple, np.ndarray)):
                inds["stretch_5"].append(bb_5[0])
                inds["%B_5"].append(bb_5[1])
                inds["bandwidth_5"].append(bb_5[2])
            else:
                inds["stretch_5"].append(np.nan)
                inds["%B_5"].append(np.nan)
                inds["bandwidth_5"].append(np.nan)   
            macd_sig_hist = ind.signal_and_histogram()
            if isinstance(macd_sig_hist, (list, tuple, np.ndarray)):
                inds["MACD_hist"].append(macd_sig_hist[1] / rv)
            else:
                inds["MACD_hist"].append(np.nan)
            inds["RSI_fast"].append(ind.RSI(7))
            inds["RSI_slow"].append(ind.RSI(21))
            inds["Dir_Ind_Diff"].append(ind.POS_DI() - ind.NEG_DI())
            inds["BOP"].append(ind.BOP(smoothed=False))
            inds["delta_BOP"].append(ind.delta_BOP(smoothed=False))
            inds["smoothed_BOP"].append(ind.BOP())
            inds["delta_sm_BOP"].append(ind.delta_BOP())
            inds["MOM_1"].append(ind.MOM(1) / rv)
            inds["MOM_3"].append(ind.MOM(3) / rv)
            inds["MOM_5"].append(ind.MOM(5) / rv)
            inds["MOM_10"].append(ind.MOM(10) / rv)
            inds["VWAP_res"].append((ind.close[i] - ind.VWAP()) / rv)
            inds["KAMA_res"].append((ind.close[i] - ind.KAMA()) / rv)
            inds["realized_vol_60"].append(sigma_60)
            inds["bid_range"].append(ind.high[i] - ind.low[i])
            for j in range(1, LAG + 1):
                inds[f"ADOSC_lag_{j}"].append(ind.ADOSC(time=i-j))
                inds[f"RSI_lag_{j}"].append(ind.RSI(time=i-j))
                inds[f"AROON_lag_{j}"].append(ind.AROON(time=i-j))
            inds["vol_ratio"].append(sigma / rv_60)
            inds["delta_KAMA_res"].append(((ind.close[i] - ind.KAMA(time=i)) 
                                           - (ind.close[i-1] - ind.KAMA(time=i-1))) 
                                           / rv)
            inds["delta_VWAP_res"].append(((ind.close[i] - ind.VWAP(time=i)) 
                                           - (ind.close[i-1] - ind.VWAP(time=i-1))) 
                                           / rv)
            inds["5m_forward_returns"].append(np.nan if np.isnan(ind.realized_vol())
                                            else ind.forward_return(5) / rv)
            pbar.update(1)
    pbar.close()

    df = df.loc[rows_kept].copy().reset_index(drop=True)
    df = df.sort_values("minute_utc").reset_index(drop=True)

    n = len(rows_kept)
    for k, v in inds.items():
        assert len(v) == n, f"{k}: {len(v)} != {n}"

    for key in inds:
        df[key] = inds[key]    

    print(df.shape)

    df.to_csv('data.csv', index=False, float_format="%.10g")

if __name__ == "__main__":
    main()
