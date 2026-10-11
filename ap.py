import streamlit as st
import requests
import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import plotly.graph_objects as go
from datetime import datetime
from scipy.signal import argrelextrema
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ==================== CONFIG ====================
st.set_page_config(page_title="Pro Crypto Dashboard v12", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #00ff88;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# ==================== AI CHATBOT SETUP ====================
try:
    from groq import Groq
    _groq_ok = True
except Exception:
    _groq_ok = False
try:
    _groq_key = st.secrets.get("GROQ_API_KEY", None)
except Exception:
    _groq_key = None

# ==================== DATA LAYER (BINANCE PRIMARY / MEXC FALLBACK) ====================
BINANCE = "https://fapi.binance.com"
MEXC = "https://contract.mexc.com"

def _get(url, params=None, timeout=8, retries=2):
    """HTTP GET with retry + backoff (handles 429 rate limits)."""
    last = None
    for a in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 418) and a < retries:
                time.sleep(0.6 * (a + 1)); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if a < retries: time.sleep(0.4 * (a + 1))
    raise last

@st.cache_data(ttl=30)
def fetch_tickers():
    try:
        data = _get(f"{BINANCE}/fapi/v1/ticker/24hr")
        rows = []
        for d in data:
            s = d['symbol']
            if not s.endswith('USDT'): continue
            if any(x in s for x in ('UPUSDT','DOWNUSDT','BULLUSDT','BEARUSDT')): continue
            rows.append({'symbol': s, 'lastPrice': float(d['lastPrice']),
                         'riseFallRate': float(d['priceChangePercent']),
                         'volume24': float(d['quoteVolume'])})
        df = pd.DataFrame(rows)
        if not df.empty:
            return df.sort_values('volume24', ascending=False).reset_index(drop=True)
    except Exception:
        pass
    try:
        d = _get(f"{MEXC}/api/v1/contract/ticker")
        if d.get("success"):
            df = pd.DataFrame(d["data"])
            df = df[df['symbol'].str.endswith('_USDT')].copy()
            for c in ['riseFallRate', 'lastPrice', 'volume24']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df['riseFallRate'] = df['riseFallRate'] * 100
            df['symbol'] = df['symbol'].str.replace('_USDT', 'USDT')
            return df.sort_values('volume24', ascending=False).reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_klines(symbol, interval="1h", limit=300):
    """Closed candles ONLY (forming candle dropped -> signals don't repaint)."""
    now_ms = int(time.time() * 1000)
    try:
        data = _get(f"{BINANCE}/fapi/v1/klines",
                    params={"symbol": symbol, "interval": interval, "limit": min(limit + 5, 1500)})
        df = pd.DataFrame([{
            'time': pd.to_datetime(k[0], unit='ms'), 'open': float(k[1]),
            'high': float(k[2]), 'low': float(k[3]), 'close': float(k[4]),
            'volume': float(k[5]), 'close_time': k[6]} for k in data])
        df = df[df['close_time'] < now_ms]
        if not df.empty:
            return df.tail(limit).reset_index(drop=True)
    except Exception:
        pass
    try:
        mi = {"15m": "Min15", "1h": "Min60", "4h": "Hour4"}.get(interval, "Min60")
        d = _get(f"{MEXC}/api/v1/contract/kline/{symbol.replace('USDT','_USDT')}",
                 params={"interval": mi})
        if d.get("success") and d.get("data"):
            k = d["data"]
            df = pd.DataFrame({'time': pd.to_datetime(k['time'], unit='s'),
                'open': pd.to_numeric(k['open']), 'high': pd.to_numeric(k['high']),
                'low': pd.to_numeric(k['low']), 'close': pd.to_numeric(k['close']),
                'volume': pd.to_numeric(k['vol'])})
            return df.iloc[:-1].tail(limit).reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_funding(symbol):
    try:
        d = _get(f"{BINANCE}/fapi/v1/premiumIndex", params={"symbol": symbol})
        return float(d["lastFundingRate"]) * 100
    except Exception:
        pass
    try:
        d = _get(f"{MEXC}/api/v1/contract/funding_rate/{symbol.replace('USDT','_USDT')}")
        if d.get("success"): return float(d["data"]["fundingRate"]) * 100
    except Exception:
        pass
    return None

@st.cache_data(ttl=60)
def fetch_taker_volume(symbol, period="1h", limit=30):
    try:
        d = _get(f"{BINANCE}/futures/data/takerlongshortRatio",
                 params={"symbol": symbol, "period": period, "limit": limit})
        if d:
            df = pd.DataFrame(d)
            df['buyVol'] = pd.to_numeric(df['buyVol']); df['sellVol'] = pd.to_numeric(df['sellVol'])
            df['CVD'] = df['buyVol'] - df['sellVol']
            return df
    except Exception:
        pass
    try:
        ccy = symbol.replace('USDT', '')
        d = _get("https://www.okx.com/api/v5/rubik/stat/taker-volume",
                 params={"ccy": ccy, "instType": "CONTRACTS", "period": "1H"})
        if d.get('code') == '0' and d.get('data'):
            df = pd.DataFrame(d['data'], columns=['timestamp', 'sellVol', 'buyVol'])
            df['buyVol'] = pd.to_numeric(df['buyVol']); df['sellVol'] = pd.to_numeric(df['sellVol'])
            df['CVD'] = df['buyVol'] - df['sellVol']
            return df
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_oi(symbol, period="1h", limit=30):
    try:
        d = _get(f"{BINANCE}/futures/data/openInterestHist",
                 params={"symbol": symbol, "period": period, "limit": limit})
        if d:
            df = pd.DataFrame(d)
            df['oi'] = pd.to_numeric(df['sumOpenInterest'])
            return df
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_sentiment(symbol):
    try:
        g = _get(f"{BINANCE}/futures/data/globalLongShortAccountRatio",
                 params={"symbol": symbol, "period": "1h", "limit": 1})
        t = _get(f"{BINANCE}/futures/data/topLongShortPositionRatio",
                 params={"symbol": symbol, "period": "1h", "limit": 1})
        if g and t:
            return {'retail_ls': float(g[0]['longShortRatio']), 'top_ls': float(t[0]['longShortRatio'])}
    except Exception:
        pass
    return None

def fetch_all_details(SYM):
    """ALL data for one coin in PARALLEL (fixes slow/only-BTC issue)."""
    with ThreadPoolExecutor(max_workers=7) as ex:
        futs = {
            'kl15': ex.submit(fetch_klines, SYM, "15m", 300),
            'kl1h': ex.submit(fetch_klines, SYM, "1h", 300),
            'kl4h': ex.submit(fetch_klines, SYM, "4h", 300),
            'funding': ex.submit(fetch_funding, SYM),
            'tv': ex.submit(fetch_taker_volume, SYM),
            'oi': ex.submit(fetch_oi, SYM),
            'sent': ex.submit(fetch_sentiment, SYM),
        }
        return {k: f.result() for k, f in futs.items()}

# ==================== TECHNICAL ANALYSIS ====================
def _last(v):
    try:
        if v is None or len(v) == 0: return None
        x = float(v.iloc[-1])
        return None if pd.isna(x) else x
    except Exception:
        return None

def calc_rsi_series(df, n=14):
    try: return ta.rsi(df['close'], length=n)
    except Exception: return None

def calc_rsi(df, n=14): return _last(calc_rsi_series(df, n))

def calc_macd(df):
    try:
        m = ta.macd(df['close'])
        if m is not None and len(m) > 0:
            return {'macd': float(m.iloc[-1, 0]), 'signal': float(m.iloc[-1, 1]), 'hist': float(m.iloc[-1, 2])}
    except Exception: pass
    return None

def calc_atr_series(df, n=14):
    try: return ta.atr(df['high'], df['low'], df['close'], length=n)
    except Exception: return None

def calc_atr(df, n=14): return _last(calc_atr_series(df, n))

def calc_adx(df, n=14):
    try:
        a = ta.adx(df['high'], df['low'], df['close'], length=n)
        return float(a.iloc[-1, 0]) if a is not None and len(a) > 0 else None
    except Exception: return None

def calc_ema(df, n=200): return _last(ta.ema(df['close'], length=n))

def calc_vp(df, bins=50):
    try:
        if len(df) < 10: return None
        pmin, pmax = df['low'].min(), df['high'].max()
        if pmin == pmax: return None
        edges = np.linspace(pmin, pmax, bins + 1)
        d = df.copy()
        d['bin'] = pd.cut(d['close'], bins=edges, labels=False)
        vp = d.groupby('bin')['volume'].sum()
        poc_bin = vp.idxmax()
        poc = (edges[poc_bin] + edges[poc_bin + 1]) / 2
        total = vp.sum(); sv = vp.sort_values(ascending=False)
        cum = 0; vb = []
        for b, v in sv.items():
            cum += v; vb.append(b)
            if cum >= total * 0.7: break
        return {'POC': poc, 'VAH': edges[max(vb) + 1], 'VAL': edges[min(vb)]}
    except Exception: return None

def calc_absorption(df):
    try:
        if len(df) < 20: return None
        d = df.copy(); d['range'] = d['high'] - d['low']
        r = d.tail(10)
        av = r['volume'].mean(); ar = r['range'].mean()
        return float(((r['volume'] / av) / (r['range'] / ar + 1e-10)).mean())
    except Exception: return None

# ==================== REGIME & DIVERGENCE ====================
def market_regime(df):
    out = {'adx': None, 'ema200': None, 'trend': 0, 'choppy': True}
    try:
        if df is None or len(df) < 60: return out
        out['adx'] = calc_adx(df)
        out['ema200'] = calc_ema(df, 200)
        if out['ema200']:
            out['trend'] = 1 if df['close'].iloc[-1] > out['ema200'] else -1
        out['choppy'] = (out['adx'] is None) or (out['adx'] < 20)
    except Exception: pass
    return out

def detect_rsi_divergence(df, rsi_ser, lookback=40, order=3):
    res = {'bull_div': False, 'bear_div': False}
    try:
        if rsi_ser is None or len(df) < lookback: return res
        d = df.tail(lookback).reset_index(drop=True)
        r = rsi_ser.tail(lookback).reset_index(drop=True)
        lows = d['low'].values; highs = d['high'].values
        pl = [i for i in argrelextrema(lows, np.less_equal, order=order)[0] if i < len(d) - 2]
        ph = [i for i in argrelextrema(highs, np.greater_equal, order=order)[0] if i < len(d) - 2]
        if len(pl) >= 2:
            i1, i2 = pl[-2], pl[-1]
            if i2 - i1 >= 3 and lows[i2] < lows[i1] and r.iloc[i2] > r.iloc[i1]:
                res['bull_div'] = True
        if len(ph) >= 2:
            i1, i2 = ph[-2], ph[-1]
            if i2 - i1 >= 3 and highs[i2] > highs[i1] and r.iloc[i2] < r.iloc[i1]:
                res['bear_div'] = True
    except Exception: pass
    return res

# ==================== SMART MONEY CONCEPTS ====================
def get_trend(df, k=3):
    if df is None or len(df) < 50: return 0, "N/A"
    try:
        highs = df['high'].values; lows = df['low'].values
        ph = argrelextrema(highs, np.greater_equal, order=k)[0]
        pl = argrelextrema(lows, np.less_equal, order=k)[0]
        if len(ph) < 2 or len(pl) < 2: return 0, "RANGE"
        lh, ph_prev = highs[ph[-1]], highs[ph[-2]]
        ll, pl_prev = lows[pl[-1]], lows[pl[-2]]
        if lh > ph_prev and ll > pl_prev: return 1, "UP"
        if lh < ph_prev and ll < pl_prev: return -1, "DOWN"
        return 0, "RANGE"
    except Exception: return 0, "N/A"

def detect_ms(df, k=3):
    res = {'bos': 0, 'choch': 0, 'status': 'No Structure'}
    if df is None or len(df) < 50: return res
    try:
        highs = df['high'].values; lows = df['low'].values; close = df['close'].values
        ph = argrelextrema(highs, np.greater_equal, order=k)[0]
        pl = argrelextrema(lows, np.less_equal, order=k)[0]
        if len(ph) < 2 or len(pl) < 2: return res
        last_ph = highs[ph[-1]]; prev_ph = highs[ph[-2]]
        last_pl = lows[pl[-1]]; prev_pl = lows[pl[-2]]
        cur = close[-1]
        up = last_ph > prev_ph and last_pl > prev_pl
        dn = last_ph < prev_ph and last_pl < prev_pl
        if up:
            if cur > last_ph: res['bos'] = 1; res['status'] = 'Bullish BoS'
            elif cur < last_pl: res['choch'] = -1; res['status'] = 'Bearish CHoCH'
        elif dn:
            if cur < last_pl: res['bos'] = -1; res['status'] = 'Bearish BoS'
            elif cur > last_ph: res['choch'] = 1; res['status'] = 'Bullish CHoCH'
        else: res['status'] = 'Ranging'
    except Exception: pass
    return res

def detect_trendline(df, window=5):
    res = {'breakout': 0, 'status': 'No Trendline', 'details': []}
    try:
        d = df.copy().reset_index(drop=True)
        highs = d['high'].values; lows = d['low'].values; close = d['close'].values
        ph = argrelextrema(highs, np.greater_equal, order=window)[0]
        pl = argrelextrema(lows, np.less_equal, order=window)[0]
        if len(ph) < 2 and len(pl) < 2: return res
        cur_idx = len(d) - 1; cur_close = close[-1]; prev_close = close[-2]
        if len(ph) >= 2:
            i1, i2 = ph[-2], ph[-1]
            if i2 > i1:
                slope = (highs[i2] - highs[i1]) / (i2 - i1)
                intercept = highs[i2] - slope * i2
                tl_cur = slope * cur_idx + intercept
                tl_prev = slope * (cur_idx - 1) + intercept
                if prev_close <= tl_prev and cur_close > tl_cur:
                    res['breakout'] = 1; res['status'] = 'Bullish Trendline Breakout'
                    res['details'] = [f"Broke above: ${tl_cur:,.4f}"]; return res
        if len(pl) >= 2:
            i1, i2 = pl[-2], pl[-1]
            if i2 > i1:
                slope = (lows[i2] - lows[i1]) / (i2 - i1)
                intercept = lows[i2] - slope * i2
                tl_cur = slope * cur_idx + intercept
                tl_prev = slope * (cur_idx - 1) + intercept
                if prev_close >= tl_prev and cur_close < tl_cur:
                    res['breakout'] = -1; res['status'] = 'Bearish Trendline Breakout'
                    res['details'] = [f"Broke below: ${tl_cur:,.4f}"]; return res
        res['status'] = 'No Trendline Breakout'
    except Exception: pass
    return res

def detect_liquidity_sweep(df, lookback=20):
    res = {'sweep': 0, 'status': 'No Sweep', 'details': []}
    try:
        if df is None or len(df) < lookback + 5: return res
        d = df.copy().reset_index(drop=True); recent = d.tail(lookback)
        for i in range(len(d) - 5, len(d)):
            c = d.iloc[i]
            body = abs(c['close'] - c['open']); rng = c['high'] - c['low']
            if rng == 0: continue
            lw = min(c['open'], c['close']) - c['low']
            uw = c['high'] - max(c['open'], c['close'])
            if lw > 2 * body and lw / rng > 0.6:
                prev_low = recent['low'].iloc[:-5].min() if len(recent) > 5 else c['low']
                if c['low'] < prev_low:
                    res['sweep'] = 1; res['status'] = 'Bullish Liquidity Sweep'; return res
            if uw > 2 * body and uw / rng > 0.6:
                prev_high = recent['high'].iloc[:-5].max() if len(recent) > 5 else c['high']
                if c['high'] > prev_high:
                    res['sweep'] = -1; res['status'] = 'Bearish Liquidity Sweep'; return res
    except Exception: pass
    return res

def detect_obs(df, lookback=50):
    obs = []
    if df is None or len(df) < 20: return obs
    try:
        d = df.tail(lookback).reset_index(drop=True)
        body = (d['close'] - d['open']).abs()
        atr_v = body.rolling(14).mean().iloc[-1]
        if pd.isna(atr_v): atr_v = body.mean()
        for i in range(2, len(d) - 2):
            if d['close'].iloc[i] < d['open'].iloc[i]:
                if (d['close'].iloc[i + 1] > d['high'].iloc[i] and
                        (d['close'].iloc[i + 2] - d['open'].iloc[i + 2]) > 1.5 * atr_v):
                    obs.append({'type': 'bullish', 'top': d['high'].iloc[i], 'bottom': d['low'].iloc[i]})
            if d['close'].iloc[i] > d['open'].iloc[i]:
                if (d['close'].iloc[i + 1] < d['low'].iloc[i] and
                        (d['open'].iloc[i + 2] - d['close'].iloc[i + 2]) > 1.5 * atr_v):
                    obs.append({'type': 'bearish', 'top': d['high'].iloc[i], 'bottom': d['low'].iloc[i]})
        return obs[-3:]
    except Exception: return []

def detect_fvg(df, lookback=50):
    fvgs = []
    if df is None or len(df) < 5: return fvgs
    try:
        d = df.tail(lookback).reset_index(drop=True)
        for i in range(2, len(d)):
            if d['low'].iloc[i] > d['high'].iloc[i - 2]:
                fvgs.append({'type': 'bullish', 'top': d['low'].iloc[i], 'bottom': d['high'].iloc[i - 2]})
            if d['high'].iloc[i] < d['low'].iloc[i - 2]:
                fvgs.append({'type': 'bearish', 'top': d['low'].iloc[i - 2], 'bottom': d['high'].iloc[i]})
        return fvgs[-3:]
    except Exception: return []

def liq_zones(df, k=5):
    zones = []
    if df is None or len(df) < 30: return zones
    try:
        d = df.tail(100).reset_index(drop=True)
        highs = d['high'].values; lows = d['low'].values
        ph = argrelextrema(highs, np.greater_equal, order=k)[0]
        pl = argrelextrema(lows, np.less_equal, order=k)[0]
        for idx in ph[-3:]: zones.append({'type': 'buy_side', 'price': float(highs[idx])})
        for idx in pl[-3:]: zones.append({'type': 'sell_side', 'price': float(lows[idx])})
        return zones
    except Exception: return []

# ==================== PERSISTENCE & HOLD TIME ====================
def check_signal_persistence(df, timeframe_min=60, min_hours=2, max_hours=10):
    try:
        if df is None or len(df) < 20:
            return {'persistence': 0, 'candles': 0, 'up_candles': 0, 'down_candles': 0,
                    'direction': 0, 'status': 'Insufficient Data', 'hours_checked': 0}
        candles_min = max(2, int(min_hours * 60 / timeframe_min))
        candles_max = min(len(df) - 5, int(max_hours * 60 / timeframe_min))
        if candles_max < candles_min: candles_max = candles_min
        recent = df.tail(candles_max).copy()
        recent['dir'] = np.sign(recent['close'] - recent['open']).astype(int)
        up = int((recent['dir'] == 1).sum()); dn = int((recent['dir'] == -1).sum())
        if up > dn * 1.5: direction = 1; persistence = up / candles_max
        elif dn > up * 1.5: direction = -1; persistence = dn / candles_max
        else: direction = 0; persistence = max(up, dn) / candles_max
        status = "Strong" if persistence > 0.65 else ("Moderate" if persistence > 0.55 else "Weak")
        return {'persistence': round(persistence, 2), 'candles': candles_max, 'up_candles': up,
                'down_candles': dn, 'direction': direction, 'status': status,
                'hours_checked': round(candles_max * timeframe_min / 60, 1)}
    except Exception:
        return {'persistence': 0, 'candles': 0, 'direction': 0, 'status': 'Error',
                'up_candles': 0, 'down_candles': 0, 'hours_checked': 0}

def estimate_hold_time(df, atr, signal_direction, vp, price):
    try:
        if df is None or len(df) < 30 or atr is None or atr == 0 or signal_direction == 0:
            return {'min_hours': 2, 'max_hours': 8, 'target_price': price, 'reason': 'Default'}
        recent = df.tail(20)
        avg_range = (recent['high'] - recent['low']).mean()
        if signal_direction == 1:
            target = vp['VAH'] if vp and vp['VAH'] > price else df['high'].tail(50).max()
            distance = target - price
        else:
            target = vp['VAL'] if vp and vp['VAL'] < price else df['low'].tail(50).min()
            distance = price - target
        if distance <= 0 or avg_range == 0:
            return {'min_hours': 2, 'max_hours': 8, 'target_price': price, 'reason': 'Invalid distance'}
        est_hours = distance / avg_range
        min_h = max(2, int(est_hours * 0.7)); max_h = max(min_h + 2, int(est_hours * 1.5))
        return {'min_hours': min_h, 'max_hours': max_h, 'target_price': float(target), 'reason': 'Volatility Based'}
    except Exception:
        return {'min_hours': 2, 'max_hours': 8, 'target_price': price, 'reason': 'Error'}

# ==================== SIGNAL ENGINE v12 (STABLE) ====================
def make_signal(price, rsi, macd, funding, cvd, vp, absorb, oi_ch, tf, ms, obs, fvgs,
                liqz, tlb, sweep, atr, persistence, regime, div, capital, risk_pct,
                swing_low=None, swing_high=None):
    bull = 0.0; bear = 0.0; reasons = []; warnings = []; details = []

    strong_trend = (not regime.get('choppy')) and regime.get('adx') is not None and regime['adx'] >= 25
    trend_up = regime.get('trend') == 1
    trend_dn = regime.get('trend') == -1
    if rsi:
        if rsi < 30 and not trend_dn: bull += 3; reasons.append("RSI Oversold")
        elif rsi < 40 and not trend_dn: bull += 1.5
        if rsi > 70 and not trend_up: bear += 3; reasons.append("RSI Overbought")
        elif rsi > 60 and not trend_up: bear += 1.5
        if trend_up and 50 < rsi < 70: bull += 1; reasons.append("Bullish Momentum")
        if trend_dn and 30 < rsi < 50: bear += 1; reasons.append("Bearish Momentum")
        details.append(f"RSI={rsi:.1f}")

    if macd:
        if macd['hist'] > 0 and macd['macd'] > macd['signal']: bull += 2; reasons.append("MACD Bullish")
        elif macd['hist'] < 0 and macd['macd'] < macd['signal']: bear += 2; reasons.append("MACD Bearish")

    if funding is not None:
        if funding > 0.05: bear += 2; reasons.append("High Funding (crowded longs)")
        elif funding < -0.05: bull += 2; reasons.append("Neg Funding (crowded shorts)")
        details.append(f"Funding={funding:.4f}%")

    if cvd is not None:
        if cvd > 0: bull += 2; reasons.append("Taker Buyers (CVD+)")
        else: bear += 2; reasons.append("Taker Sellers (CVD-)")
        details.append(f"CVD={cvd:,.0f}")

    if vp and price:
        if price < vp['VAL']: bull += 0.5 if trend_dn else 2
        if price < vp['VAL'] and not trend_dn: reasons.append("Below Value Area (discount)")
        if price > vp['VAH']: bear += 0.5 if trend_up else 2
        if price > vp['VAH'] and not trend_up: reasons.append("Above Value Area (premium)")
        if vp['VAL'] <= price <= vp['VAH']:
            if price < vp['POC']: bull += 0.5
            else: bear += 0.5
        details.append(f"VP: POC={vp['POC']:.4f}")

    if absorb and absorb > 1.5: reasons.append("High Absorption")

    if oi_ch is not None:
        if oi_ch > 2: bull += 1.5; reasons.append("OI Increasing")
        elif oi_ch < -2: bear += 1.5; reasons.append("OI Decreasing")

    t15 = tf.get('15m', (0, 'N/A'))[0]; t1h = tf.get('1H', (0, 'N/A'))[0]; t4h = tf.get('4H', (0, 'N/A'))[0]
    if t15 == 1 and t1h == 1 and t4h == 1: bull += 3; reasons.append("MTF All Bullish")
    elif t15 == -1 and t1h == -1 and t4h == -1: bear += 3; reasons.append("MTF All Bearish")
    details.append(f"MTF: 15m={tf['15m'][1]} 1H={tf['1H'][1]} 4H={tf['4H'][1]}")

    if ms['bos'] == 1: bull += 2.5; reasons.append("Bullish BoS")
    elif ms['bos'] == -1: bear += 2.5; reasons.append("Bearish BoS")
    elif ms['choch'] == 1: bull += 2; reasons.append("Bullish CHoCH")
    elif ms['choch'] == -1: bear += 2; reasons.append("Bearish CHoCH")
    details.append(f"Structure: {ms['status']}")

    if obs and price:
        if any(o['type'] == 'bullish' and o['bottom'] <= price <= o['top'] for o in obs):
            bull += 2; reasons.append("In Bullish OB")
        if any(o['type'] == 'bearish' and o['bottom'] <= price <= o['top'] for o in obs):
            bear += 2; reasons.append("In Bearish OB")

    if fvgs and price:
        if any(f['type'] == 'bullish' and f['bottom'] <= price <= f['top'] for f in fvgs):
            bull += 1.5; reasons.append("In Bullish FVG")
        if any(f['type'] == 'bearish' and f['bottom'] <= price <= f['top'] for f in fvgs):
            bear += 1.5; reasons.append("In Bearish FVG")

    if liqz and price:
        for z in liqz:
            if abs(price - z['price']) / price < 0.005:
                if z['type'] == 'sell_side': bull += 1; reasons.append("Near Sell-Side Liq")
                else: bear += 1; reasons.append("Near Buy-Side Liq")

    if tlb['breakout'] == 1: bull += 3; reasons.append("Bullish Trendline Breakout")
    elif tlb['breakout'] == -1: bear += 3; reasons.append("Bearish Trendline Breakout")

    if sweep['sweep'] == 1: bull += 2.5; reasons.append("Bullish Liquidity Sweep")
    elif sweep['sweep'] == -1: bear += 2.5; reasons.append("Bearish Liquidity Sweep")

    if div.get('bull_div'): bull += 2.5; reasons.append("Bullish RSI Divergence")
    if div.get('bear_div'): bear += 2.5; reasons.append("Bearish RSI Divergence")

    if persistence and persistence['candles'] > 0:
        if persistence['direction'] == 1 and persistence['persistence'] > 0.6:
            bull += 2; reasons.append(f"Bullish Persistence {persistence['persistence']*100:.0f}%/{persistence['hours_checked']}h")
        elif persistence['direction'] == -1 and persistence['persistence'] > 0.6:
            bear += 2; reasons.append(f"Bearish Persistence {persistence['persistence']*100:.0f}%/{persistence['hours_checked']}h")

    # regime
    if regime.get('choppy'):
        warnings.append("Choppy market (ADX < 20) — new trades risky")
    if regime.get('ema200') and regime.get('adx') is not None:
        details.append(f"EMA200=${regime['ema200']:,.4f} | ADX={regime['adx']:.1f}")

    # probability split
    total = bull + bear
    if total == 0: bp, sp, hp = 33.3, 33.3, 33.4
    else:
        bp = (bull / total) * 100; sp = (bear / total) * 100; hp = max(0, 100 - bp - sp)
    if bp > sp and bp > 50: bp = min(95, bp + 10); sp = max(5, sp - 5)
    elif sp > bp and sp > 50: sp = min(95, sp + 10); bp = max(5, bp - 5)
    hp = max(0, 100 - bp - sp)

    margin = abs(bp - sp)
    strong_buy = bull >= 12 and bull >= bear * 2
    strong_sell = bear >= 12 and bear >= bull * 2
    min_confluence = 6.0

    if strong_buy: signal, conf = "STRONG BUY 🚀", bp
    elif strong_sell: signal, conf = "STRONG SELL 🔻", sp
    elif bp >= 55 and bull >= min_confluence and margin >= 15: signal, conf = "BUY", bp
    elif sp >= 55 and bear >= min_confluence and margin >= 15: signal, conf = "SELL", sp
    elif bp >= 55 or sp >= 55: signal, conf = "WAIT (No Edge) 🟡", hp; warnings.append(f"Margin too small ({margin:.0f}% < 15%) — market 50/50 hai, wait karo")
    else: signal, conf = "WAIT / HOLD", hp

    if "BUY" in signal and regime.get('trend') == -1 and not strong_buy:
        warnings.append("Counter-trend long (price below EMA200)")
    if "SELL" in signal and regime.get('trend') == 1 and not strong_sell:
        warnings.append("Counter-trend short (price above EMA200)")

    # SL / TP (structure-aware, R:R gated)
    entry = price
    atr_v = atr if atr else entry * 0.02
    sl = tp1 = tp2 = tp3 = None; rr1 = None; size_usdt = size_units = None

    if "BUY" in signal:
        sl = (swing_low - 0.25 * atr_v) if swing_low else entry - 1.5 * atr_v
        sl = min(sl, entry - 1.2 * atr_v)   # at least 1.2 ATR away
        sl = max(sl, entry - 2.5 * atr_v)   # at most 2.5 ATR away
        cands = [z['price'] for z in (liqz or []) if z['type'] == 'buy_side' and z['price'] > entry + 1.0 * atr_v]
        if vp and vp['VAH'] > entry + 1.0 * atr_v: cands.append(vp['VAH'])
        tp1 = min(cands) if cands else entry + 2.5 * atr_v
        risk = entry - sl
        tp2 = entry + 2 * risk; tp3 = entry + 3 * risk
    elif "SELL" in signal:
        sl = (swing_high + 0.25 * atr_v) if swing_high else entry + 1.5 * atr_v
        sl = max(sl, entry + 1.2 * atr_v)
        sl = min(sl, entry + 2.5 * atr_v)
        cands = [z['price'] for z in (liqz or []) if z['type'] == 'sell_side' and z['price'] < entry - 1.0 * atr_v]
        if vp and vp['VAL'] < entry - 1.0 * atr_v: cands.append(vp['VAL'])
        tp1 = max(cands) if cands else entry - 2.5 * atr_v
        risk = sl - entry
        tp2 = entry - 2 * risk; tp3 = entry - 3 * risk

    if sl and tp1 and abs(entry - sl) > 1e-12:
        rr1 = abs(tp1 - entry) / abs(entry - sl)
        details.append(f"R:R = 1:{rr1:.2f}")
        if regime.get('choppy') and not (strong_buy or strong_sell):
            signal = "WAIT (Choppy) ⏸️"; warnings.append("Signal blocked by regime filter")
        elif rr1 < 1.5 and not (strong_buy or strong_sell):
            signal = "WATCHLIST ⚠️"; warnings.append(f"Poor R:R (1:{rr1:.2f}) — better entry ka wait karo")
        risk_amt = capital * risk_pct / 100.0
        size_units = risk_amt / abs(entry - sl)
        size_usdt = size_units * entry

    return {'signal': signal, 'conf': conf, 'bp': round(bp, 1), 'sp': round(sp, 1), 'hp': round(hp, 1),
            'bull': round(bull, 1), 'bear': round(bear, 1), 'margin': round(margin, 1),
            'reasons': reasons, 'warnings': warnings, 'details': details,
            'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
            'rr1': round(rr1, 2) if rr1 else None,
            'size_usdt': round(size_usdt, 2) if size_usdt else None,
            'size_units': round(size_units, 4) if size_units else None,
            'strong_buy': strong_buy, 'strong_sell': strong_sell}

# ==================== BACKTESTING ENGINE ====================
def run_backtest(df, min_score=6.0, sl_mult=1.5, tp_mult=2.5, max_hold=48, vp_window=120):
    """
    Walk-forward backtest on 1H closed candles.
    Simplified confluence: RSI + MACD + EMA200 trend + ADX regime + Volume Profile.
    (Full SMC stack vectorize nahi hota — yeh indicator-based approximation hai.)
    Conservative rule: same candle mein SL+TP dono hit ho to SL count hota hai.
    """
    d = df.reset_index(drop=True).copy()
    n = len(d)
    if n < 260:
        return None
    close = d['close'].values; open_ = d['open'].values
    high = d['high'].values; low = d['low'].values; vol = d['volume'].values

    rsi = ta.rsi(d['close'], length=14).values
    macd_df = ta.macd(d['close'])
    macd_l = macd_df.iloc[:, 0].values; macd_s = macd_df.iloc[:, 1].values; macd_h = macd_df.iloc[:, 2].values
    atr = ta.atr(d['high'], d['low'], d['close'], length=14).values
    adx_df = ta.adx(d['high'], d['low'], d['close'], length=14)
    adx = adx_df.iloc[:, 0].values
    ema200 = ta.ema(d['close'], length=200).values

    def vp_levels(idx):
        w_close = close[idx - vp_window:idx]; w_vol = vol[idx - vp_window:idx]
        if len(w_close) < 20 or w_close.max() == w_close.min(): return None
        hist, edges = np.histogram(w_close, bins=40, weights=w_vol)
        poc_i = int(np.argmax(hist))
        poc = (edges[poc_i] + edges[poc_i + 1]) / 2
        order = np.argsort(hist)[::-1]
        cum = 0; tot = hist.sum(); bins_sel = []
        for b in order:
            cum += hist[b]; bins_sel.append(b)
            if cum >= 0.7 * tot: break
        return {'POC': poc, 'VAH': edges[max(bins_sel) + 1], 'VAL': edges[min(bins_sel)]}

    def score_at(i):
        b = s_ = 0.0
        r = rsi[i]
        trend_up = (not np.isnan(ema200[i])) and close[i] > ema200[i]
        trend_dn = (not np.isnan(ema200[i])) and close[i] < ema200[i]
        strong = (not np.isnan(adx[i])) and adx[i] >= 25
        if not np.isnan(r):
            # RSI mean-reversion sirf AGAINST trend mein — trend ke saath to continuation hai
            if r < 30 and not trend_dn: b += 3
            elif r < 40 and not trend_dn: b += 1.5
            if r > 70 and not trend_up: s_ += 3
            elif r > 60 and not trend_up: s_ += 1.5
            if trend_up and 50 < r < 70: b += 1.5
            if trend_dn and 30 < r < 50: s_ += 1.5
        if not (np.isnan(macd_h[i]) or np.isnan(macd_l[i]) or np.isnan(macd_s[i])):
            if macd_h[i] > 0 and macd_l[i] > macd_s[i]: b += 2
            elif macd_h[i] < 0 and macd_l[i] < macd_s[i]: s_ += 2
        if np.isnan(adx[i]) or adx[i] < 20 or np.isnan(ema200[i]):
            return 0, b, s_  # choppy / warmup -> no trade
        if trend_up: b += 2.0
        else: s_ += 2.0
        v = vp_levels(i)
        if v:
            if close[i] < v['VAL']: b += 0.5 if trend_dn else 2
            elif close[i] > v['VAH']: s_ += 0.5 if trend_up else 2
            elif close[i] < v['POC']: b += 0.5
            else: s_ += 0.5
        return 1, b, s_

    trades = []
    equity = [0.0]
    start = max(205, vp_window + 5)
    i = start
    while i < n - 2:
        ok, b, s_ = score_at(i)
        direction = 0
        if ok:
            if b >= min_score and b >= s_ * 1.3 and (b - s_) >= 3: direction = 1
            elif s_ >= min_score and s_ >= b * 1.3 and (s_ - b) >= 3: direction = -1
        if direction == 0:
            i += 1; continue
        ei = i + 1
        if np.isnan(atr[i]) or atr[i] == 0: i += 1; continue
        entry = open_[ei]
        a = atr[i]
        if direction == 1:
            sl_p = entry - sl_mult * a; tp_p = entry + tp_mult * a
        else:
            sl_p = entry + sl_mult * a; tp_p = entry - tp_mult * a
        risk = abs(entry - sl_p)
        j = ei; exit_p = None; r_mult = None
        while j < n and j <= ei + max_hold:
            if direction == 1:
                if low[j] <= sl_p: exit_p = sl_p; r_mult = -1.0; break
                if high[j] >= tp_p: exit_p = tp_p; r_mult = (tp_p - entry) / risk; break
            else:
                if high[j] >= sl_p: exit_p = sl_p; r_mult = -1.0; break
                if low[j] <= tp_p: exit_p = tp_p; r_mult = (entry - tp_p) / risk; break
            j += 1
        if exit_p is None:
            j = min(j, n - 1)
            exit_p = close[j]
            r_mult = ((exit_p - entry) if direction == 1 else (entry - exit_p)) / risk
        equity.append(equity[-1] + r_mult)
        trades.append({'#': len(trades) + 1, 'Dir': 'LONG' if direction == 1 else 'SHORT',
                       'Entry Time': d['time'].iloc[ei], 'Entry': round(entry, 6),
                       'SL': round(sl_p, 6), 'TP': round(tp_p, 6), 'Exit': round(exit_p, 6),
                       'Bars': j - ei, 'R': round(r_mult, 2)})
        i = j + 1  # one position at a time

    if not trades: return {'trades': [], 'n': 0}
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf['R'] > 0]; losses = tdf[tdf['R'] <= 0]
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    max_dd = float((peak - eq).max()) if len(eq) else 0.0
    gross_w = wins['R'].sum() if len(wins) else 0.0
    gross_l = abs(losses['R'].sum()) if len(losses) else 0.0
    return {
        'trades': tdf, 'equity': eq,
        'n': len(tdf), 'win_rate': round(len(wins) / len(tdf) * 100, 1),
        'avg_r': round(tdf['R'].mean(), 2), 'total_r': round(tdf['R'].sum(), 2),
        'profit_factor': round(gross_w / gross_l, 2) if gross_l > 0 else float('inf'),
        'max_dd_r': round(max_dd, 2),
        'avg_bars': round(tdf['Bars'].mean(), 1),
        'expectancy': round(tdf['R'].mean(), 2)}

# ==================== PARALLEL SCANNER ====================
def analyze_symbol(sym, tickers_df, deep=False):
    """Light mode: klines only (fast). Deep mode: + funding/CVD/OI."""
    try:
        df1h = fetch_klines(sym, "1h", 300)
        if df1h.empty or len(df1h) < 60:
            return ('error', sym, 'no klines')
        price = float(df1h['close'].iloc[-1])
        rsi_ser = calc_rsi_series(df1h)
        rsi = _last(rsi_ser)
        macd = calc_macd(df1h); vp = calc_vp(df1h); atr = calc_atr(df1h)
        funding = cvd = oi_ch = None
        if deep:
            try: funding = fetch_funding(sym)
            except Exception: pass
            try:
                tv = fetch_taker_volume(sym)
                if not tv.empty: cvd = float(tv['CVD'].iloc[-1])
            except Exception: pass
            try:
                oidf = fetch_oi(sym)
                if not oidf.empty and len(oidf) > 1 and oidf['oi'].iloc[0]:
                    oi_ch = ((oidf['oi'].iloc[-1] - oidf['oi'].iloc[0]) / oidf['oi'].iloc[0]) * 100
            except Exception: pass
        df15 = fetch_klines(sym, "15m", 300)
        df4h = fetch_klines(sym, "4h", 300)
        tf = {'15m': get_trend(df15), '1H': get_trend(df1h), '4H': get_trend(df4h)}
        ms = detect_ms(df1h); obs = detect_obs(df1h); fvgs = detect_fvg(df1h)
        liqz = liq_zones(df1h); tlb = detect_trendline(df1h); sweep = detect_liquidity_sweep(df1h)
        persist = check_signal_persistence(df1h, 60, 2, 10)
        regime = market_regime(df1h)
        div = detect_rsi_divergence(df1h, rsi_ser)
        swing_low = float(df1h['low'].tail(15).min())
        swing_high = float(df1h['high'].tail(15).max())
        sig = make_signal(price, rsi, macd, funding, cvd, vp, None, oi_ch, tf, ms, obs,
                          fvgs, liqz, tlb, sweep, atr, persist, regime, div, 1000.0, 1.0,
                          swing_low, swing_high)
        row = tickers_df[tickers_df['symbol'] == sym].iloc[0]
        base = {'Symbol': sym, 'Price': f"${row['lastPrice']:,.4f}",
                '24h %': f"{row['riseFallRate']:+.2f}%", 'Vol ($M)': f"{row['volume24']/1e6:.1f}",
                'Signal': sig['signal'], 'Bull': sig['bull'], 'Bear': sig['bear'],
                'ADX': f"{regime['adx']:.1f}" if regime['adx'] else '-',
                'R:R': f"1:{sig['rr1']}" if sig['rr1'] else '-',
                'Entry': f"${sig['entry']:,.4f}",
                'SL': f"${sig['sl']:,.4f}" if sig['sl'] else '-',
                'TP1': f"${sig['tp1']:,.4f}" if sig['tp1'] else '-'}
        if "BUY" in sig['signal'] and sig['bull'] >= 8:
            return ('buy', base, None)
        if "SELL" in sig['signal'] and sig['bear'] >= 8:
            return ('sell', base, None)
        return ('none', None, None)
    except Exception as e:
        return ('error', sym, str(e)[:80])

def run_scan(symbols, tickers_df, deep=False, max_workers=15):
    buy_rows, sell_rows, errors = [], [], []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(analyze_symbol, s, tickers_df, deep): s for s in symbols}
        for fut in as_completed(futs):
            done += 1
            yield ('progress', done, len(symbols))
            try:
                kind, a, b = fut.result()
            except Exception as e:
                errors.append((futs[fut], str(e)[:80])); continue
            if kind == 'buy': buy_rows.append(a)
            elif kind == 'sell': sell_rows.append(a)
            elif kind == 'error': errors.append((a, b))
    buy_rows.sort(key=lambda x: x['Bull'], reverse=True)
    sell_rows.sort(key=lambda x: x['Bear'], reverse=True)
    yield ('done', buy_rows, sell_rows, errors)

# ==================== MODE SELECTOR ====================
st.sidebar.markdown("## 🧭 Mode Selector")
MODE = st.sidebar.radio("Kaunsa mode?", ["📊 Crypto Dashboard", "🧪 Backtesting Lab", "🤖 AI Chatbot"], index=0)

if MODE == "🤖 AI Chatbot":
    st.title("🤖 AI Chatbot")
    if not _groq_ok or not _groq_key:
        st.error("groq library ya GROQ_API_KEY missing hai. Secrets configure karein.")
        st.stop()
    client = Groq(api_key=_groq_key)
    st.sidebar.header("⚙️ Chatbot Settings")
    model = st.sidebar.selectbox("Model", [
        "openai/gpt-oss-120b", "openai/gpt-oss-20b",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "qwen/qwen3-32b", "moonshotai/kimi-k2-instruct"], index=0)
    temp = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    sys_prompt = st.sidebar.text_area("System Prompt", "You are a helpful AI assistant.", height=100)
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []; st.rerun()
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if prompt := st.chat_input("Apna sawal likho..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            ph = st.empty(); full = ""
            try:
                stream = client.chat.completions.create(
                    model=model, temperature=temp, stream=True,
                    messages=[{"role": "system", "content": sys_prompt}] + st.session_state.messages)
                for ch in stream:
                    if ch.choices[0].delta.content:
                        full += ch.choices[0].delta.content
                        ph.markdown(full + "▌")
                ph.markdown(full)
            except Exception as e:
                ph.error(f"Error: {e}")
        st.session_state.messages.append({"role": "assistant", "content": full})
    st.stop()

# ==================== BACKTESTING LAB ====================
if MODE == "🧪 Backtesting Lab":
    st.title("🧪 Backtesting Lab")
    st.caption("Strategy ka matlab: RSI + MACD + EMA200 + ADX regime + Volume Profile confluence | 1H closed candles | SL-first conservative")
    tickers_bt = fetch_tickers()
    if tickers_bt.empty:
        st.error("Data fetch nahi hua."); st.stop()
    bt_syms = tickers_bt['symbol'].tolist()
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        b_sym = st.selectbox("Symbol", bt_syms, index=bt_syms.index('BTCUSDT') if 'BTCUSDT' in bt_syms else 0)
    with b2:
        b_hist = st.selectbox("History (1H candles)", [500, 1000, 1500], index=1)
    with b3:
        b_min_score = st.number_input("Min Score", 4.0, 12.0, 6.0, 0.5)
    with b4:
        b_max_hold = st.number_input("Max Hold (bars)", 12, 96, 48, 4)

    if st.button("▶ Run Backtest", type="primary"):
        with st.spinner(f"{b_sym} ka data load ho raha hai..."):
            df = fetch_klines(b_sym, "1h", b_hist)
        if df.empty or len(df) < 260:
            st.error("Kaafi data nahi mila (kam az kam 260 candles chahiye)."); st.stop()
        with st.spinner("Backtest chal raha hai... (30-60 sec)"):
            res = run_backtest(df, b_min_score, 1.5, 2.5, b_max_hold)
        if res is None or res['n'] == 0:
            st.warning("Is setting pe koi trade nahi bana. Min Score kam kar ke dobara try karein.")
            st.stop()
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total Trades", res['n'])
        k2.metric("Win Rate", f"{res['win_rate']}%")
        k3.metric("Avg R", f"{res['avg_r']:+.2f}R")
        k4.metric("Total R", f"{res['total_r']:+.2f}R")
        k5.metric("Profit Factor", res['profit_factor'])
        k6.metric("Max Drawdown", f"{res['max_dd_r']}R")
        st.caption(f"Avg hold: {res['avg_bars']:.1f} bars (~{res['avg_bars']:.0f} ghante) | ⚠️ Fees/slippage shamil nahi — real results ~10-20% kam honge. Ye proof nahi, screening tool hai.")

        fig_eq = go.Figure(go.Scatter(y=res['equity'], mode='lines', name='Equity (R)',
                                      line=dict(color='#00ff88')))
        fig_eq.update_layout(height=300, template="plotly_dark", title="Equity Curve (R multiples)",
                             xaxis_title="Trade #", yaxis_title="Cumulative R")
        st.plotly_chart(fig_eq, use_container_width=True)

        with st.expander(f"📋 All Trades ({res['n']})"):
            st.dataframe(res['trades'], use_container_width=True, hide_index=True)

        if res['win_rate'] < 40 or res['profit_factor'] < 1.2:
            st.error("⚠️ Ye settings is coin pe kaam nahi kar rahe. Coin ya timeframe change karein, ya min score barhayein.")
        elif res['profit_factor'] >= 1.5:
            st.success("✅ Settings promising lag rahe hain — Lekin paper trade ke baghair real mat karna!")
        st.stop()

# ==================== DASHBOARD MODE ====================
st.title("📊 Professional Crypto Dashboard v12")
st.caption(f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Closed-candle signals | Flip filter | Regime filter | Parallel data")

CAPITAL = st.sidebar.number_input("💰 Capital ($)", min_value=10.0, value=1000.0, step=100.0)
RISK_PCT = st.sidebar.number_input("⚠️ Risk per trade (%)", min_value=0.1, value=1.0, step=0.25)
st.sidebar.caption("Risk 1-2% recommended.")

tickers = fetch_tickers()
if tickers.empty:
    st.error("Data fetch nahi hua. Internet check karein.")
    st.stop()
sorted_df = tickers.sort_values('riseFallRate', ascending=False)
all_sym = sorted_df['symbol'].tolist()

# ==================== SCANNER ====================
with st.expander("🎯 Smart Signal Scanner (Parallel — Fast)", expanded=True):
    s1, s2, s3, s4 = st.columns(4)
    with s1: sc_top = st.selectbox("Top coins", [30, 50, 100, 200], index=1)
    with s2: sc_min_pts = st.number_input("Min Points", 5.0, 20.0, 8.0, 0.5)
    with s3: sc_min_vol = st.number_input("Min Vol ($M)", 0.0, 500.0, 20.0, 5.0)
    with s4: sc_deep = st.checkbox("Deep scan (funding/CVD/OI — slow)", value=False)

    if st.button("🚀 Scan (Parallel)", type="primary"):
        scan_df = sorted_df[sorted_df['volume24'] / 1e6 >= sc_min_vol].head(sc_top)
        symbols = scan_df['symbol'].tolist()
        if not symbols: st.warning("Koi coin filter pass nahi hua.")
        else:
            progress = st.progress(0); status = st.empty()
            buy_rows, sell_rows, errors = [], [], []
            for item in run_scan(symbols, scan_df, deep=sc_deep):
                if item[0] == 'done':
                    _, buy_rows, sell_rows, errors = item
                else:
                    _, d, t = item
                    progress.progress(d / t)
                    status.text(f"Scanning... {d}/{t}")
            status.text(f"✅ Done! {len(symbols)} scanned | 🟢 {len(buy_rows)} buys | 🔴 {len(sell_rows)} sells | ❌ {len(errors)} failed")
            if errors:
                with st.expander(f"❌ Failed symbols ({len(errors)}) — kyun fail hue"):
                    st.dataframe(pd.DataFrame(errors, columns=['Symbol', 'Reason']), use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                if buy_rows: st.subheader(f"🟢 Strong BUY ({len(buy_rows)})")
                st.dataframe(pd.DataFrame(buy_rows), use_container_width=True, hide_index=True)
            with c2:
                if sell_rows: st.subheader(f"🔴 Strong SELL ({len(sell_rows)})")
                st.dataframe(pd.DataFrame(sell_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ==================== COIN SELECTOR ====================
st.markdown("### 🔍 Coin Select Karo (Detailed Analysis)")
c1, c2 = st.columns([2, 2])
with c1:
    di = all_sym.index('BTCUSDT') if 'BTCUSDT' in all_sym else 0
    sel_dd = st.selectbox("Dropdown:", all_sym, index=di)
with c2:
    sel_txt = st.text_input("Ya likho (e.g. ETHUSDT):")

SYM = sel_txt.strip().upper() if sel_txt.strip().upper() in all_sym else sel_dd
st.markdown(f"### ✅ Selected: `{SYM}`")

# ---- FLIP FILTER state (resets when coin changes) ----
if 'ff_coin' not in st.session_state or st.session_state.ff_coin != SYM:
    st.session_state.ff_coin = SYM
    st.session_state.ff_dir = 0
    st.session_state.ff_label = ""

crow = tickers[tickers['symbol'] == SYM]
vol_m = 0.0
if not crow.empty:
    co = crow.iloc[0]; vol_m = co['volume24'] / 1e6
    m1, m2, m3 = st.columns(3)
    m1.metric("Price (live)", f"${co['lastPrice']:,.4f}", f"{co['riseFallRate']:.2f}%")
    m2.metric("24h Volume", f"${co['volume24']:,.0f}")
    m3.metric("Volume ($M)", f"${vol_m:.1f}M")
    if vol_m < 20: st.warning(f"⚠️ LOW VOLUME: ${vol_m:.1f}M. Trade mat karo.")

# ---- PARALLEL data fetch (fixes slow/only-BTC) ----
with st.spinner(f"{SYM} ka data load ho raha hai..."):
    data = fetch_all_details(SYM)

kl15, kl1h, kl4h = data['kl15'], data['kl1h'], data['kl4h']
funding = data['funding']
tv = data['tv']; cvd = float(tv['CVD'].iloc[-1]) if not tv.empty else None
oidf = data['oi']
oi_ch = None
if not oidf.empty and len(oidf) > 1 and oidf['oi'].iloc[0]:
    oi_ch = ((oidf['oi'].iloc[-1] - oidf['oi'].iloc[0]) / oidf['oi'].iloc[0]) * 100
sent = data['sent']

if kl1h.empty:
    st.warning("1H data nahi mila — Binance aur MEXC dono se fail. Thodi der baad refresh karein.")
    st.stop()

price = float(kl1h['close'].iloc[-1])  # CLOSED candle — stable (no flicker)
rsi_ser = calc_rsi_series(kl1h); rsi = _last(rsi_ser)
macd = calc_macd(kl1h); vp = calc_vp(kl1h); absorb = calc_absorption(kl1h)
atr = calc_atr(kl1h)

tf_trends = {'15m': get_trend(kl15), '1H': get_trend(kl1h), '4H': get_trend(kl4h)}
ms = detect_ms(kl1h); obs = detect_obs(kl1h); fvgs = detect_fvg(kl1h)
liqz = liq_zones(kl1h); tlb = detect_trendline(kl1h); sweep = detect_liquidity_sweep(kl1h)
persist = check_signal_persistence(kl1h, 60, 2, 10)
regime = market_regime(kl1h)
div = detect_rsi_divergence(kl1h, rsi_ser)
swing_low = float(kl1h['low'].tail(15).min())
swing_high = float(kl1h['high'].tail(15).max())

sig = make_signal(price, rsi, macd, funding, cvd, vp, absorb, oi_ch, tf_trends, ms,
                  obs, fvgs, liqz, tlb, sweep, atr, persist, regime, div, CAPITAL, RISK_PCT,
                  swing_low, swing_high)

# ---- FLIP FILTER: buy->sell->buy suppression ----
raw_dir = 1 if (sig['bull'] > sig['bear'] and sig['bp'] >= 55) else (-1 if (sig['bear'] > sig['bull'] and sig['sp'] >= 55) else 0)
FLIP_MIN_MARGIN = 20.0
if raw_dir != 0 and st.session_state.ff_dir != 0 and raw_dir != st.session_state.ff_dir and sig['margin'] < FLIP_MIN_MARGIN:
    old = "BUY" if st.session_state.ff_dir == 1 else "SELL"
    sig['signal'] = f"WAIT (Flip Filter ⏸️)"
    sig['warnings'].append(f"Flip suppressed: pehle {old} tha, naya signal sirf {sig['margin']:.0f}% margin se — 20%+ hone par hi flip confirm hota hai")
    raw_dir = 0
elif raw_dir != 0:
    st.session_state.ff_dir = raw_dir
    st.session_state.ff_label = sig['signal']

bias = 1 if sig['bull'] > sig['bear'] else (-1 if sig['bear'] > sig['bull'] else 0)
hold = estimate_hold_time(kl1h, atr, bias, vp, price)

# ==================== DISPLAY ====================
st.header(f"🎯 Signal: {SYM}")
p1, p2, p3 = st.columns(3)
p1.metric("🟢 BUY Probability", f"{sig['bp']}%")
p2.metric("🔴 SELL Probability", f"{sig['sp']}%")
p3.metric("🟡 HOLD Probability", f"{sig['hp']}%")

fig_p = go.Figure(go.Bar(x=['BUY', 'SELL', 'HOLD'], y=[sig['bp'], sig['sp'], sig['hp']],
    marker_color=['#00ff88', '#ff4444', '#ffaa00'],
    text=[f"{sig['bp']}%", f"{sig['sp']}%", f"{sig['hp']}%"], textposition='auto'))
fig_p.update_layout(height=220, template="plotly_dark", showlegend=False, yaxis_range=[0, 100])
st.plotly_chart(fig_p, use_container_width=True)

if tlb['breakout'] != 0: st.warning(f"⚠️ **{tlb['status']}**")
if sweep['sweep'] != 0: st.info(f"🎯 **{sweep['status']}**")
if div.get('bull_div'): st.success("📈 **Bullish RSI Divergence detected**")
if div.get('bear_div'): st.error("📉 **Bearish RSI Divergence detected**")
for w in sig['warnings']: st.warning(f"⛔ {w}")

f1, f2 = st.columns([1, 1])
with f1:
    if "STRONG BUY" in sig['signal']: st.success(f"## 🚀 {sig['signal']}")
    elif "STRONG SELL" in sig['signal']: st.error(f"## 🔻 {sig['signal']}")
    elif "BUY" in sig['signal']: st.success(f"## 🟢 {sig['signal']}")
    elif "SELL" in sig['signal']: st.error(f"## 🔴 {sig['signal']}")
    elif "WATCHLIST" in sig['signal']: st.warning(f"## ⚠️ {sig['signal']}")
    else: st.warning(f"## 🟡 {sig['signal']}")
    st.metric("Confidence", f"{sig['conf']:.1f}%", f"Margin {sig['margin']:.0f}%")
with f2:
    st.metric("Analysis Price (closed candle)", f"${sig['entry']:,.4f}")
    if sig['sl']: st.metric("Stop Loss", f"${sig['sl']:,.4f}")

if sig['tp1']:
    t1, t2, t3 = st.columns(3)
    t1.metric("TP1", f"${sig['tp1']:,.4f}"); t2.metric("TP2", f"${sig['tp2']:,.4f}"); t3.metric("TP3", f"${sig['tp3']:,.4f}")

if sig['size_usdt']:
    st.header("💼 Position Sizing")
    s1, s2, s3 = st.columns(3)
    s1.metric("Risk Amount", f"${CAPITAL * RISK_PCT / 100:.2f}", f"{RISK_PCT}% of ${CAPITAL:,.0f}")
    s2.metric("Position Size", f"{sig['size_units']:,} {SYM.replace('USDT','')}")
    s3.metric("Position Value", f"${sig['size_usdt']:,.2f}")
    st.caption(f"R:R = 1:{sig['rr1']} | SL touch hone par sirf ${CAPITAL * RISK_PCT / 100:.2f} loss.")

# ---- HOLD TIME (ab hamesha dikhega jab bias ho) ----
st.header("⏳ Signal Quality & Hold Time")
q1, q2, q3 = st.columns(3)
q1.metric("Persistence", persist['status'], f"{persist['persistence']*100:.0f}%")
q2.metric("Hours Analyzed", f"{persist['hours_checked']}h", f"{persist['up_candles']}↑ / {persist['down_candles']}↓")
if bias != 0 and atr:
    q3.metric("Est. Hold Time", f"{hold['min_hours']}h - {hold['max_hours']}h",
              f"Target: ${hold['target_price']:,.4f}")
else:
    q3.metric("Est. Hold Time", "N/A", "koi clear bias nahi")
st.caption(f"Basis: {hold['reason']} | Flip filter: active (min 20% margin for direction change)")

st.header("🛡️ Regime Filter")
r1, r2, r3 = st.columns(3)
r1.metric("ADX", f"{regime['adx']:.1f}" if regime['adx'] else "N/A", "Trend strength")
r2.metric("EMA200", f"${regime['ema200']:,.4f}" if regime['ema200'] else "N/A")
r3.metric("Market Regime", "CHOPPY ⏸️" if regime['choppy'] else ("TRENDING UP 📈" if regime['trend'] == 1 else "TRENDING DOWN 📉"))
if regime['choppy']: st.warning("⚠️ ADX < 20 — range market. Naye trades ka win-rate low hota hai.")

if sent:
    st.header("👥 Market Sentiment (Binance)")
    se1, se2 = st.columns(2)
    se1.metric("Retail Long/Short", f"{sent['retail_ls']:.3f}", ">1 = zyada longs")
    se2.metric("Top Traders L/S", f"{sent['top_ls']:.3f}", ">1 = zyada longs")

with st.expander("📋 Full Analysis Details"):
    st.markdown(f"**Bull Points:** {sig['bull']} | **Bear Points:** {sig['bear']} | **Margin:** {sig['margin']}%")
    for d in sig['details']: st.write(f"• {d}")
    st.markdown("**Confluence Reasons:**")
    for r in sig['reasons']: st.write(f"✅ {r}")
    if sig['warnings']:
        st.markdown("**Warnings:**")
        for w in sig['warnings']: st.write(f"⛔ {w}")

st.header("⏱️ MTF Trends")
mt1, mt2, mt3 = st.columns(3)
mt1.metric("15m", tf_trends['15m'][1]); mt2.metric("1H", tf_trends['1H'][1]); mt3.metric("4H", tf_trends['4H'][1])

st.header("🏗️ Market Structure & SMC")
st.info(f"**Structure:** {ms['status']}")
if obs:
    st.dataframe(pd.DataFrame([{"Type": o['type'], "Top": f"${o['top']:,.4f}", "Bottom": f"${o['bottom']:,.4f}"} for o in obs]),
                 use_container_width=True, hide_index=True)
if fvgs:
    st.dataframe(pd.DataFrame([{"Type": f['type'], "Top": f"${f['top']:,.4f}", "Bottom": f"${f['bottom']:,.4f}"} for f in fvgs]),
                 use_container_width=True, hide_index=True)

st.header("📊 Indicators")
i1, i2, i3, i4 = st.columns(4)
with i1:
    if rsi: st.metric("RSI", f"{rsi:.1f}")
with i2:
    if macd: st.metric("MACD Hist", f"{macd['hist']:.4f}")
with i3:
    if funding is not None: st.metric("Funding", f"{funding:.4f}%")
with i4:
    if cvd is not None: st.metric("Taker CVD", f"{cvd:,.0f}")

if vp:
    v1, v2, v3 = st.columns(3)
    v1.metric("POC", f"${vp['POC']:,.4f}"); v2.metric("VAH", f"${vp['VAH']:,.4f}"); v3.metric("VAL", f"${vp['VAL']:,.4f}")

st.header(f"📈 {SYM} (1H — closed candles)")
fig = go.Figure(data=[go.Candlestick(x=kl1h['time'], open=kl1h['open'],
    high=kl1h['high'], low=kl1h['low'], close=kl1h['close'])])
if regime.get('ema200'): fig.add_hline(y=regime['ema200'], line_dash="dash", line_color="orange", annotation_text="EMA200")
if vp:
    fig.add_hline(y=vp['POC'], line_dash="dash", line_color="yellow", annotation_text="POC")
    fig.add_hline(y=vp['VAH'], line_dash="dot", line_color="green", annotation_text="VAH")
    fig.add_hline(y=vp['VAL'], line_dash="dot", line_color="red", annotation_text="VAL")
if sig['sl'] and sig['tp1'] and ("BUY" in sig['signal'] or "SELL" in sig['signal']):
    fig.add_hline(y=sig['sl'], line_dash="solid", line_color='red', annotation_text="SL")
    fig.add_hline(y=sig['tp1'], line_dash="solid", line_color="#00ff88", annotation_text="TP1")
fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.header("✅ v12 Fixes")
st.sidebar.markdown("""
1. **Parallel data fetch** — ab sab coins equally fast
2. **Flip filter** — BUY→SELL palatna band (min 20% margin)
3. **Hold time fix** — ab bias ke saath hamesha dikhta hai
4. **Closed-candle price** — signal flicker nahi karta
5. **Backtesting Lab** — alag section
6. **Scanner errors visible** — kaunsa coin kyun fail hua
7. **Retry logic** — 429/rate-limit handling
""")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
