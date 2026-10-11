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
st.set_page_config(page_title="Pro Crypto Dashboard v11", layout="wide", page_icon="📊")

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

def _get(url, params=None, timeout=10):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=30)
def fetch_tickers():
    """Binance Futures 24h tickers. Fallback: MEXC."""
    try:
        data = _get(f"{BINANCE}/fapi/v1/ticker/24hr")
        rows = []
        for d in data:
            s = d['symbol']
            if not s.endswith('USDT'):
                continue
            if any(x in s for x in ('UPUSDT','DOWNUSDT','BULLUSDT','BEARUSDT')):
                continue  # leveraged tokens skip
            rows.append({'symbol': s, 'lastPrice': float(d['lastPrice']),
                         'riseFallRate': float(d['priceChangePercent']),
                         'volume24': float(d['quoteVolume'])})
        df = pd.DataFrame(rows)
        if not df.empty:
            return df.sort_values('volume24', ascending=False).reset_index(drop=True)
    except Exception:
        pass
    try:  # MEXC fallback
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
    """Closed candles ONLY (last forming candle dropped -> no repaint)."""
    now_ms = int(time.time() * 1000)
    try:  # Binance
        data = _get(f"{BINANCE}/fapi/v1/klines",
                    params={"symbol": symbol, "interval": interval, "limit": limit + 5})
        df = pd.DataFrame([{
            'time': pd.to_datetime(k[0], unit='ms'), 'open': float(k[1]),
            'high': float(k[2]), 'low': float(k[3]), 'close': float(k[4]),
            'volume': float(k[5]), 'close_time': k[6]} for k in data])
        df = df[df['close_time'] < now_ms]  # closed only
        if not df.empty:
            return df.tail(limit).reset_index(drop=True)
    except Exception:
        pass
    try:  # MEXC fallback
        mi = {"15m": "Min15", "1h": "Min60", "4h": "Hour4"}.get(interval, "Min60")
        d = _get(f"{MEXC}/api/v1/contract/kline/{symbol.replace('USDT','_USDT')}",
                 params={"interval": mi})
        if d.get("success") and d.get("data"):
            k = d["data"]
            df = pd.DataFrame({'time': pd.to_datetime(k['time'], unit='s'),
                'open': pd.to_numeric(k['open']), 'high': pd.to_numeric(k['high']),
                'low': pd.to_numeric(k['low']), 'close': pd.to_numeric(k['close']),
                'volume': pd.to_numeric(k['vol'])})
            return df.iloc[:-1].tail(limit).reset_index(drop=True)  # drop forming
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
        if d.get("success"):
            return float(d["data"]["fundingRate"]) * 100
    except Exception:
        pass
    return None

@st.cache_data(ttl=60)
def fetch_taker_volume(symbol, period="1h", limit=30):
    """Taker buy/sell volume (CVD proxy). Fallback: OKX."""
    try:
        d = _get(f"{BINANCE}/futures/data/takerlongshortRatio",
                 params={"symbol": symbol, "period": period, "limit": limit})
        if d:
            df = pd.DataFrame(d)
            df['buyVol'] = pd.to_numeric(df['buyVol'])
            df['sellVol'] = pd.to_numeric(df['sellVol'])
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
    """Retail vs top-trader long/short positioning."""
    try:
        g = _get(f"{BINANCE}/futures/data/globalLongShortAccountRatio",
                 params={"symbol": symbol, "period": "1h", "limit": 1})
        t = _get(f"{BINANCE}/futures/data/topLongShortPositionRatio",
                 params={"symbol": symbol, "period": "1h", "limit": 1})
        if g and t:
            return {'retail_ls': float(g[0]['longShortRatio']),
                    'top_ls': float(t[0]['longShortRatio'])}
    except Exception:
        pass
    return None

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

def calc_rsi(df, n=14):
    return _last(calc_rsi_series(df, n))

def calc_macd(df):
    try:
        m = ta.macd(df['close'])
        if m is not None and len(m) > 0:
            return {'macd': float(m.iloc[-1, 0]), 'signal': float(m.iloc[-1, 1]), 'hist': float(m.iloc[-1, 2])}
    except Exception: pass
    return None

def calc_atr(df, n=14):
    try: return _last(ta.atr(df['high'], df['low'], df['close'], length=n))
    except Exception: return None

def calc_adx(df, n=14):
    try:
        a = ta.adx(df['high'], df['low'], df['close'], length=n)
        return float(a.iloc[-1, 0]) if a is not None and len(a) > 0 else None
    except Exception: return None

def calc_ema(df, n=200):
    try: return _last(ta.ema(df['close'], length=n))
    except Exception: return None

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

# ==================== REGIME FILTER (chop detection) ====================
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

# ==================== SMART MONEY CONCEPTS (closed candles) ====================
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
        res['status'] = 'No Trendline Breakout (Skipped)'
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
        if df is None or len(df) < 30 or atr is None or atr == 0:
            return {'min_hours': 2, 'max_hours': 8, 'target_price': price, 'reason': 'Default'}
        recent = df.tail(20)
        avg_range = (recent['high'] - recent['low']).mean()
        if signal_direction == 1:
            target = vp['VAH'] if vp and vp['VAH'] > price else df['high'].tail(50).max()
            distance = target - price
        elif signal_direction == -1:
            target = vp['VAL'] if vp and vp['VAL'] < price else df['low'].tail(50).min()
            distance = price - target
        else:
            return {'min_hours': 2, 'max_hours': 8, 'target_price': price, 'reason': 'No signal'}
        if distance <= 0 or avg_range == 0:
            return {'min_hours': 2, 'max_hours': 8, 'target_price': price, 'reason': 'Invalid distance'}
        est_hours = distance / avg_range
        min_h = max(2, int(est_hours * 0.7)); max_h = max(min_h + 2, int(est_hours * 1.5))
        return {'min_hours': min_h, 'max_hours': max_h, 'target_price': float(target), 'reason': 'Volatility Based'}
    except Exception:
        return {'min_hours': 2, 'max_hours': 8, 'target_price': price, 'reason': 'Error'}

# ==================== SIGNAL ENGINE v2 ====================
def make_signal(price, rsi, macd, funding, cvd, vp, absorb, oi_ch, tf, ms, obs, fvgs,
                liqz, tlb, sweep, atr, persistence, regime, div, capital, risk_pct):
    bull = 0.0; bear = 0.0; reasons = []; warnings = []
    details = []

    if rsi:
        if rsi < 30: bull += 3; reasons.append("RSI Oversold")
        elif rsi < 40: bull += 1.5
        elif rsi > 70: bear += 3; reasons.append("RSI Overbought")
        elif rsi > 60: bear += 1.5
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
        if price < vp['VAL']: bull += 2; reasons.append("Below Value Area (discount)")
        elif price > vp['VAH']: bear += 2; reasons.append("Above Value Area (premium)")
        elif price < vp['POC']: bull += 0.5
        else: bear += 0.5
        details.append(f"VP: POC={vp['POC']:.4f}")

    if absorb and absorb > 1.5: reasons.append("High Absorption")

    if oi_ch is not None:
        if oi_ch > 2: bull += 1.5; reasons.append("OI Increasing (new positions)")
        elif oi_ch < -2: bear += 1.5; reasons.append("OI Decreasing (positions closing)")

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

    # --- v2 additions: divergence ---
    if div.get('bull_div'): bull += 2.5; reasons.append("Bullish RSI Divergence")
    if div.get('bear_div'): bear += 2.5; reasons.append("Bearish RSI Divergence")

    # --- v2 additions: persistence ---
    if persistence and persistence['candles'] > 0:
        if persistence['direction'] == 1 and persistence['persistence'] > 0.6:
            bull += 2; reasons.append(f"Bullish Persistence {persistence['persistence']*100:.0f}%/{persistence['hours_checked']}h")
        elif persistence['direction'] == -1 and persistence['persistence'] > 0.6:
            bear += 2; reasons.append(f"Bearish Persistence {persistence['persistence']*100:.0f}%/{persistence['hours_checked']}h")

    # --- v2 additions: regime filter ---
    if regime.get('choppy'):
        warnings.append(f"Choppy market (ADX={regime.get('adx', 0):.1f} < 20) — new trades risky")
    if regime.get('ema200'):
        details.append(f"EMA200=${regime['ema200']:,.4f} | ADX={regime.get('adx', 0):.1f}")

    # probability split
    total = bull + bear
    if total == 0: bp, sp, hp = 33.3, 33.3, 33.4
    else:
        bp = (bull / total) * 100; sp = (bear / total) * 100; hp = max(0, 100 - bp - sp)
    if bp > sp and bp > 50: bp = min(95, bp + 10); sp = max(5, sp - 5)
    elif sp > bp and sp > 50: sp = min(95, sp + 10); bp = max(5, bp - 5)
    hp = max(0, 100 - bp - sp)

    strong_buy = bull >= 12 and bull >= bear * 2
    strong_sell = bear >= 12 and bear >= bull * 2
    if strong_buy: signal, conf = "STRONG BUY 🚀", bp
    elif strong_sell: signal, conf = "STRONG SELL 🔻", sp
    elif bp >= 55: signal, conf = "BUY", bp
    elif sp >= 55: signal, conf = "SELL", sp
    else: signal, conf = "WAIT / HOLD", hp

    # --- v2: counter-trend penalty ---
    if "BUY" in signal and regime.get('trend') == -1 and not strong_buy:
        bear += 1.5; warnings.append("Counter-trend long (price below EMA200)")
    if "SELL" in signal and regime.get('trend') == 1 and not strong_sell:
        bull += 1.5; warnings.append("Counter-trend short (price above EMA200)")

    # --- v2: structure-based SL / TP with R:R enforcement ---
    entry = price
    atr_v = atr if atr else entry * 0.02
    recent_lows = None; recent_highs = None
    sl = tp1 = tp2 = tp3 = None; rr1 = None
    size_usdt = size_units = None

    if "BUY" in signal:
        swing_low = entry * 0.98  # fallback
        sl_struct = swing_low - 0.25 * atr_v
        sl_atr = entry - 1.5 * atr_v
        sl = max(min(sl_struct, entry - 0.8 * atr_v), entry - 2.5 * atr_v)
        sl = min(sl, sl_atr) if sl > sl_atr else sl
        sl = min(sl, entry - 1.2 * atr_v)  # never tighter than 1.2 ATR
        # TP1: nearest buy-side liquidity / VAH above, else 2.5 ATR
        cands = [z['price'] for z in (liqz or []) if z['type'] == 'buy_side' and z['price'] > entry + 1.0 * atr_v]
        if vp and vp['VAH'] > entry + 1.0 * atr_v: cands.append(vp['VAH'])
        tp1 = min(cands) if cands else entry + 2.5 * atr_v
        risk = entry - sl
        tp2 = entry + 2 * risk; tp3 = entry + 3 * risk
    elif "SELL" in signal:
        swing_high = entry * 1.02
        sl_struct = swing_high + 0.25 * atr_v
        sl_atr = entry + 1.5 * atr_v
        sl = min(max(sl_struct, entry + 0.8 * atr_v), entry + 2.5 * atr_v)
        sl = max(sl, entry + 1.2 * atr_v)
        cands = [z['price'] for z in (liqz or []) if z['type'] == 'sell_side' and z['price'] < entry - 1.0 * atr_v]
        if vp and vp['VAL'] < entry - 1.0 * atr_v: cands.append(vp['VAL'])
        tp1 = max(cands) if cands else entry - 2.5 * atr_v
        risk = sl - entry
        tp2 = entry - 2 * risk; tp3 = entry - 3 * risk

    if sl and tp1 and entry != sl:
        rr1 = abs(tp1 - entry) / abs(entry - sl)
        details.append(f"R:R = 1:{rr1:.2f}")
        # --- v2: quality gates ---
        if regime.get('choppy') and not (strong_buy or strong_sell):
            signal = "WAIT (Choppy) ⏸️"; warnings.append("Signal blocked by regime filter")
        elif rr1 < 1.5 and not (strong_buy or strong_sell):
            signal = "WATCHLIST ⚠️"; warnings.append(f"Poor R:R (1:{rr1:.2f}) — wait for better entry")
        # position sizing
        risk_amt = capital * risk_pct / 100.0
        size_units = risk_amt / abs(entry - sl)
        size_usdt = size_units * entry

    return {'signal': signal, 'conf': conf, 'bp': round(bp, 1), 'sp': round(sp, 1), 'hp': round(hp, 1),
            'bull': round(bull, 1), 'bear': round(bear, 1), 'reasons': reasons, 'warnings': warnings,
            'details': details, 'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
            'rr1': round(rr1, 2) if rr1 else None,
            'size_usdt': round(size_usdt, 2) if size_usdt else None,
            'size_units': round(size_units, 4) if size_units else None,
            'strong_buy': strong_buy, 'strong_sell': strong_sell}

# ==================== PARALLEL SYMBOL ANALYSIS (for scanners) ====================
def analyze_symbol(sym, tickers_df):
    """Full pipeline for one symbol. Returns (sym, buy_row, sell_row) or None."""
    try:
        df1h = fetch_klines(sym, "1h", 300)
        if df1h.empty or len(df1h) < 60: return None
        price = float(df1h['close'].iloc[-1])
        rsi_ser = calc_rsi_series(df1h)
        rsi = _last(rsi_ser)
        macd = calc_macd(df1h); vp = calc_vp(df1h)
        atr = calc_atr(df1h); funding = fetch_funding(sym)
        tv = fetch_taker_volume(sym)
        cvd = float(tv['CVD'].iloc[-1]) if not tv.empty else None
        oidf = fetch_oi(sym)
        oi_ch = None
        if not oidf.empty and len(oidf) > 1 and oidf['oi'].iloc[0]:
            oi_ch = ((oidf['oi'].iloc[-1] - oidf['oi'].iloc[0]) / oidf['oi'].iloc[0]) * 100
        df15 = fetch_klines(sym, "15m", 300)
        df4h = fetch_klines(sym, "4h", 300)
        tf = {'15m': get_trend(df15), '1H': get_trend(df1h), '4H': get_trend(df4h)}
        ms = detect_ms(df1h); obs = detect_obs(df1h); fvgs = detect_fvg(df1h)
        liqz = liq_zones(df1h); tlb = detect_trendline(df1h); sweep = detect_liquidity_sweep(df1h)
        persist = check_signal_persistence(df1h, 60, 2, 10)
        regime = market_regime(df1h)
        div = detect_rsi_divergence(df1h, rsi_ser)
        sig = make_signal(price, rsi, macd, funding, cvd, vp, None, oi_ch, tf, ms, obs,
                          fvgs, liqz, tlb, sweep, atr, persist, regime, div, 1000.0, 1.0)
        row = tickers_df[tickers_df['symbol'] == sym].iloc[0]
        base = {'Symbol': sym, 'Price': f"${row['lastPrice']:,.4f}",
                '24h %': f"{row['riseFallRate']:+.2f}%", 'Vol ($M)': f"{row['volume24']/1e6:.1f}",
                'Signal': sig['signal'], 'Bull': sig['bull'], 'Bear': sig['bear'],
                'ADX': f"{regime['adx']:.1f}" if regime['adx'] else '-',
                'R:R': f"1:{sig['rr1']}" if sig['rr1'] else '-',
                'Entry': f"${sig['entry']:,.4f}",
                'SL': f"${sig['sl']:,.4f}" if sig['sl'] else '-',
                'TP1': f"${sig['tp1']:,.4f}" if sig['tp1'] else '-'}
        if sig['bull'] >= 8 and sig['bull'] > sig['bear'] * 1.5:
            return ('buy', base)
        if sig['bear'] >= 8 and sig['bear'] > sig['bull'] * 1.5:
            return ('sell', base)
        return None
    except Exception:
        return None

def run_scan(symbols, tickers_df, max_workers=10):
    """Parallel scan. Returns (buy_rows, sell_rows)."""
    buy_rows, sell_rows = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(analyze_symbol, s, tickers_df): s for s in symbols}
        for fut in as_completed(futs):
            done += 1
            yield done, len(symbols)  # progress
            r = fut.result()
            if r:
                side, row = r
                (buy_rows if side == 'buy' else sell_rows).append(row)
    buy_rows.sort(key=lambda x: x['Bull'], reverse=True)
    sell_rows.sort(key=lambda x: x['Bear'], reverse=True)
    yield ('done', buy_rows, sell_rows)

# ==================== MODE SELECTOR ====================
st.sidebar.markdown("## 🧭 Mode Selector")
MODE = st.sidebar.radio("Kaunsa mode?", ["📊 Crypto Dashboard", "🤖 AI Chatbot"], index=0)

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

# ==================== DASHBOARD MODE ====================
st.title("📊 Professional Crypto Dashboard v11")
st.caption(f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Closed-candle signals (no repaint) | Regime filter | R:R gate")

CAPITAL = st.sidebar.number_input("💰 Capital ($)", min_value=10.0, value=1000.0, step=100.0)
RISK_PCT = st.sidebar.number_input("⚠️ Risk per trade (%)", min_value=0.1, value=1.0, step=0.25)
st.sidebar.caption("Risk 1-2% recommended. Kabhi bhi zyada mat karo.")

tickers = fetch_tickers()
if tickers.empty:
    st.error("Data fetch nahi hua. Internet check karein.")
    st.stop()
sorted_df = tickers.sort_values('riseFallRate', ascending=False)
all_sym = sorted_df['symbol'].tolist()

# ==================== SCANNER ====================
with st.expander("🎯 Smart Signal Scanner (Parallel — Fast)", expanded=True):
    s1, s2, s3 = st.columns(3)
    with s1: sc_top = st.selectbox("Top coins", [30, 50, 100, 200], index=1)
    with s2: sc_min_pts = st.number_input("Min Points", 5.0, 20.0, 8.0, 0.5)
    with s3: sc_min_vol = st.number_input("Min Vol ($M)", 0.0, 500.0, 20.0, 5.0)

    if st.button("🚀 Scan (Parallel)", type="primary"):
        scan_df = sorted_df[sorted_df['volume24'] / 1e6 >= sc_min_vol].head(sc_top)
        symbols = scan_df['symbol'].tolist()
        if not symbols: st.warning("Koi coin filter pass nahi hua.")
        else:
            progress = st.progress(0); status = st.empty()
            buy_rows, sell_rows = [], []
            gen = run_scan(symbols, scan_df)
            for item in gen:
                if item[0] == 'done':
                    _, buy_rows, sell_rows = item
                else:
                    d, t = item
                    progress.progress(d / t)
                    status.text(f"Scanning... {d}/{t}")
            status.text(f"✅ Done! {len(symbols)} scanned | 🟢 {len(buy_rows)} buys | 🔴 {len(sell_rows)} sells")
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

crow = tickers[tickers['symbol'] == SYM]
if not crow.empty:
    co = crow.iloc[0]; vol_m = co['volume24'] / 1e6
    m1, m2, m3 = st.columns(3)
    m1.metric("Price", f"${co['lastPrice']:,.4f}", f"{co['riseFallRate']:.2f}%")
    m2.metric("24h Volume", f"${co['volume24']:,.0f}")
    m3.metric("Volume ($M)", f"${vol_m:.1f}M")
    if vol_m < 20: st.warning(f"⚠️ LOW VOLUME: ${vol_m:.1f}M. Trade mat karo.")

kl15 = fetch_klines(SYM, "15m", 300)
kl1h = fetch_klines(SYM, "1h", 300)
kl4h = fetch_klines(SYM, "4h", 300)
if kl1h.empty:
    st.warning("Data nahi mila."); st.stop()

price = float(tickers.loc[tickers['symbol'] == SYM, 'lastPrice'].iloc[0]) if not crow.empty else float(kl1h['close'].iloc[-1])
rsi_ser = calc_rsi_series(kl1h)
rsi = _last(rsi_ser)
macd = calc_macd(kl1h); vp = calc_vp(kl1h); absorb = calc_absorption(kl1h)
atr = calc_atr(kl1h); funding = fetch_funding(SYM)
tv = fetch_taker_volume(SYM)
cvd = float(tv['CVD'].iloc[-1]) if not tv.empty else None
oidf = fetch_oi(SYM)
oi_ch = None
if not oidf.empty and len(oidf) > 1 and oidf['oi'].iloc[0]:
    oi_ch = ((oidf['oi'].iloc[-1] - oidf['oi'].iloc[0]) / oidf['oi'].iloc[0]) * 100
sent = fetch_sentiment(SYM)

tf_trends = {'15m': get_trend(kl15), '1H': get_trend(kl1h), '4H': get_trend(kl4h)}
ms = detect_ms(kl1h); obs = detect_obs(kl1h); fvgs = detect_fvg(kl1h)
liqz = liq_zones(kl1h); tlb = detect_trendline(kl1h); sweep = detect_liquidity_sweep(kl1h)
persist = check_signal_persistence(kl1h, 60, 2, 10)
regime = market_regime(kl1h)
div = detect_rsi_divergence(kl1h, rsi_ser)

sig = make_signal(price, rsi, macd, funding, cvd, vp, absorb, oi_ch, tf_trends, ms,
                  obs, fvgs, liqz, tlb, sweep, atr, persist, regime, div, CAPITAL, RISK_PCT)
signal_dir = 1 if "BUY" in sig['signal'] else (-1 if "SELL" in sig['signal'] else 0)
hold = estimate_hold_time(kl1h, atr, signal_dir, vp, price)

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
    st.metric("Confidence", f"{sig['conf']:.1f}%")
with f2:
    st.metric("Entry Price", f"${sig['entry']:,.4f}")
    if sig['sl']: st.metric("Stop Loss", f"${sig['sl']:,.4f}")

if sig['tp1']:
    t1, t2, t3 = st.columns(3)
    t1.metric("TP1", f"${sig['tp1']:,.4f}"); t2.metric("TP2", f"${sig['tp2']:,.4f}"); t3.metric("TP3", f"${sig['tp3']:,.4f}")

# ---- Position sizing panel ----
if sig['size_usdt']:
    st.header("💼 Position Sizing")
    s1, s2, s3 = st.columns(3)
    s1.metric("Risk Amount", f"${CAPITAL * RISK_PCT / 100:.2f}", f"{RISK_PCT}% of ${CAPITAL:,.0f}")
    s2.metric("Position Size", f"{sig['size_units']:,} {SYM.replace('USDT','')}")
    s3.metric("Position Value", f"${sig['size_usdt']:,.2f}")
    st.caption(f"R:R = 1:{sig['rr1']} | SL ke touch hone par sirf ${CAPITAL * RISK_PCT / 100:.2f} loss — yahi discipline profitable trader banata hai.")

st.header("⏳ Signal Quality & Hold Time")
q1, q2, q3 = st.columns(3)
q1.metric("Persistence", persist['status'], f"{persist['persistence']*100:.0f}%")
q2.metric("Hours Analyzed", f"{persist['hours_checked']}h", f"{persist['up_candles']}↑ / {persist['down_candles']}↓")
q3.metric("Est. Hold Time", f"{hold['min_hours']}h - {hold['max_hours']}h" if signal_dir != 0 else "N/A",
          f"Target: ${hold['target_price']:,.4f}" if signal_dir != 0 else "")

st.header("🛡️ Regime Filter (v11)")
r1, r2, r3 = st.columns(3)
r1.metric("ADX", f"{regime['adx']:.1f}" if regime['adx'] else "N/A", "Trend strength")
r2.metric("EMA200", f"${regime['ema200']:,.4f}" if regime['ema200'] else "N/A")
r3.metric("Market Regime", "CHOPPY ⏸️" if regime['choppy'] else ("TRENDING UP 📈" if regime['trend'] == 1 else "TRENDING DOWN 📉"))
if regime['choppy']: st.warning("⚠️ ADX < 20 — market range mein hai. Is waqt naye trades ka win-rate low hota hai.")

if sent:
    st.header("👥 Market Sentiment (Binance)")
    se1, se2 = st.columns(2)
    se1.metric("Retail Long/Short", f"{sent['retail_ls']:.3f}", ">1 = zyada longs")
    se2.metric("Top Traders L/S", f"{sent['top_ls']:.3f}", ">1 = zyada longs")
    if sent['retail_ls'] > 2 and sent['top_ls'] < 1.2: st.info("Retail bullish, top traders neutral — careful with longs")
    if sent['retail_ls'] < 0.6 and sent['top_ls'] > 1.2: st.info("Retail bearish, top traders long — possible squeeze up")

with st.expander("📋 Full Analysis Details"):
    st.markdown(f"**Bull Points:** {sig['bull']} | **Bear Points:** {sig['bear']}")
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
if sig['sl'] and sig['tp1']:
    color_sl = 'red' if "BUY" in sig['signal'] else 'lime'
    fig.add_hline(y=sig['sl'], line_dash="solid", line_color=color_sl, annotation_text="SL")
    fig.add_hline(y=sig['tp1'], line_dash="solid", line_color="#00ff88", annotation_text="TP1")
fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.header("✅ v11 Improvements")
st.sidebar.markdown("""
1. **Binance Futures** primary data (reliable, liquid)
2. **Closed candles only** — no repaint
3. **Regime filter** (ADX) — chop mein trade block
4. **RSI Divergence** detection
5. **R:R gate** — bura trade auto-reject
6. **Position sizing** (risk-based)
7. **Parallel scanner** — 5-10x faster
8. **Market sentiment** (retail vs top traders)
""")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
