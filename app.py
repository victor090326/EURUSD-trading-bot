import io
import re
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="EUR/USD Trading Analysis", page_icon="📈", layout="wide")

st.title("EUR/USD Trading Analysis")
st.caption("Technical analysis workspace • Chart-image review is optional")

st.warning(
    "Educational decision support only — not financial advice. Signals are estimates, "
    "not guarantees. Confirm prices and execution with your broker before trading."
)

REQUIRED = {"open", "high", "low", "close"}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {"date": "timestamp", "datetime": "timestamp", "time": "timestamp"}
    df = df.rename(columns=aliases)
    return df

def load_ohlc(uploaded_file, pasted_text: str) -> pd.DataFrame:
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    elif pasted_text.strip():
        raw = pd.read_csv(io.StringIO(pasted_text.strip()))
    else:
        raise ValueError("Upload an OHLC CSV file or paste candle data.")

    raw = normalize_columns(raw)
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(sorted(missing)) +
            ". Required columns are open, high, low, close. A timestamp column is optional."
        )

    for col in REQUIRED:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
        raw = raw.dropna(subset=["timestamp"])
        raw = raw.sort_values("timestamp")

    raw = raw.dropna(subset=list(REQUIRED))
    valid = (
        (raw["open"] > 0) & (raw["high"] >= raw[["open", "close", "low"]].max(axis=1)) &
        (raw["low"] <= raw[["open", "close", "high"]].min(axis=1))
    )
    raw = raw.loc[valid].drop_duplicates(
        subset=["timestamp"] if "timestamp" in raw.columns else ["open", "high", "low", "close"],
        keep="last"
    ).reset_index(drop=True)

    if len(raw) < 60:
        raise ValueError(f"Only {len(raw)} valid candles found. Please provide at least 60.")

    return raw

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()

    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi14"] = 100 - (100 / (1 + rs))
    d.loc[(avg_loss == 0) & (avg_gain > 0), "rsi14"] = 100

    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    d["atr14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    # Fractals are confirmed only after two candles to the right.
    d["swing_high"] = False
    d["swing_low"] = False
    for i in range(2, len(d) - 2):
        d.loc[i, "swing_high"] = d.loc[i, "high"] > d.loc[i-2:i+2, "high"].drop(i).max()
        d.loc[i, "swing_low"] = d.loc[i, "low"] < d.loc[i-2:i+2, "low"].drop(i).min()

    d["last_swing_high"] = d["high"].where(d["swing_high"]).ffill().shift(1)
    d["last_swing_low"] = d["low"].where(d["swing_low"]).ffill().shift(1)
    d["bos"] = "NO BOS"
    bull = (d["close"] > d["last_swing_high"]) & (d["close"].shift(1) <= d["last_swing_high"])
    bear = (d["close"] < d["last_swing_low"]) & (d["close"].shift(1) >= d["last_swing_low"])
    d.loc[bull, "bos"] = "BULLISH BOS"
    d.loc[bear, "bos"] = "BEARISH BOS"
    return d

def timeframe_direction(df: pd.DataFrame, minutes: int) -> str:
    if "timestamp" not in df.columns:
        return "TIME REQUIRED"
    indexed = df.set_index("timestamp")
    candles = indexed.resample(f"{minutes}min", label="right", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    counts = indexed["close"].resample(f"{minutes}min", label="right", closed="left").count()
    candles = candles.loc[counts == minutes // 5] if minutes > 5 else candles
    if len(candles) < 50:
        return "INSUFFICIENT DATA"
    e20 = candles["close"].ewm(span=20, adjust=False).mean().iloc[-1]
    e50 = candles["close"].ewm(span=50, adjust=False).mean().iloc[-1]
    return "BULLISH" if e20 > e50 else "BEARISH" if e20 < e50 else "NEUTRAL"

def build_report(d: pd.DataFrame, timeframe: str, sl_atr: float, rr: float):
    last = d.iloc[-1]
    if pd.isna(last["rsi14"]) or pd.isna(last["atr14"]):
        raise ValueError("Indicators are not ready. Upload more candle history.")

    score = 0
    reasons = []
    if last["ema20"] > last["ema50"]:
        score += 2
        reasons.append("EMA 20 is above EMA 50 (bullish trend filter).")
    elif last["ema20"] < last["ema50"]:
        score -= 2
        reasons.append("EMA 20 is below EMA 50 (bearish trend filter).")
    else:
        reasons.append("EMA 20 and EMA 50 are equal (no trend edge).")

    if last["rsi14"] > 50:
        score += 1
        reasons.append(f"RSI is {last['rsi14']:.1f}, above 50 (positive momentum).")
    elif last["rsi14"] < 50:
        score -= 1
        reasons.append(f"RSI is {last['rsi14']:.1f}, below 50 (negative momentum).")
    else:
        reasons.append("RSI is 50 (neutral momentum).")

    if last["bos"] == "BULLISH BOS":
        score += 2
        reasons.append("Latest candle shows a bullish break of structure.")
    elif last["bos"] == "BEARISH BOS":
        score -= 2
        reasons.append("Latest candle shows a bearish break of structure.")
    else:
        reasons.append("No new break of structure on the latest candle.")

    signal = "BUY" if score >= 3 else "SELL" if score <= -3 else "NO TRADE"
    entry = float(last["close"])
    atr = float(last["atr14"])
    if signal == "BUY":
        sl, tp = entry - sl_atr * atr, entry + rr * sl_atr * atr
    elif signal == "SELL":
        sl, tp = entry + sl_atr * atr, entry - rr * sl_atr * atr
    else:
        sl = tp = None

    tf_bias = "NOT AVAILABLE"
    if timeframe == "5-minute candles" and "timestamp" in d.columns:
        dirs = {m: timeframe_direction(d, m) for m in (5, 15, 60)}
        if all(v == "BULLISH" for v in dirs.values()):
            tf_bias = "BULLISH"
        elif all(v == "BEARISH" for v in dirs.values()):
            tf_bias = "BEARISH"
        else:
            tf_bias = "MIXED / INSUFFICIENT"
    return {
        "signal": signal, "score": score, "price": entry,
        "rsi": float(last["rsi14"]), "atr": atr, "ema20": float(last["ema20"]),
        "ema50": float(last["ema50"]), "bos": last["bos"],
        "timeframe_bias": tf_bias, "stop_loss": sl, "take_profit": tp,
        "reasons": reasons
    }

with st.sidebar:
    st.header("Analysis settings")
    timeframe = st.selectbox("Candle timeframe", ["5-minute candles", "Other / unknown"])
    sl_atr = st.slider("Stop-loss distance (ATR)", 0.5, 3.0, 1.5, 0.1)
    rr = st.slider("Reward-to-risk target", 1.0, 5.0, 2.0, 0.5)
    st.caption("Multi-timeframe comparison is calculated only for timestamped 5-minute data.")

tab1, tab2 = st.tabs(["📊 Candle-data analysis", "🖼️ Chart screenshot analysis"])

with tab1:
    st.subheader("Upload or paste EUR/USD candle data")
    st.write("CSV columns required: `open, high, low, close`. Add `timestamp` for multi-timeframe analysis.")
    uploaded = st.file_uploader("Upload OHLC CSV", type=["csv"], key="ohlc")
    sample = "timestamp,open,high,low,close\n2026-01-01 00:00:00,1.1000,1.1010,1.0990,1.1005"
    pasted = st.text_area("Or paste CSV candle data", height=120, placeholder=sample)
    if st.button("Analyse candle data", type="primary", use_container_width=True):
        try:
            raw = load_ohlc(uploaded, pasted)
            analyzed = add_indicators(raw)
            report = build_report(analyzed, timeframe, sl_atr, rr)
            st.session_state["analysis_data"] = analyzed
            st.session_state["analysis_report"] = report
            st.success("Analysis completed.")
        except Exception as exc:
            st.error(str(exc))

    if "analysis_report" in st.session_state:
        r = st.session_state["analysis_report"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Potential decision", r["signal"])
        c2.metric("Signal score", f'{r["score"]:+d}')
        c3.metric("Latest close", f'{r["price"]:.5f}')
        st.subheader("Market readings")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RSI (14)", f'{r["rsi"]:.2f}')
        m2.metric("EMA 20", f'{r["ema20"]:.5f}')
        m3.metric("EMA 50", f'{r["ema50"]:.5f}')
        m4.metric("Structure", r["bos"])
        st.metric("Multi-timeframe bias", r["timeframe_bias"])
        st.subheader("Why the system reached this result")
        for reason in r["reasons"]:
            st.write("• " + reason)
        if r["signal"] != "NO TRADE":
            a, b = st.columns(2)
            a.metric("Illustrative stop loss", f'{r["stop_loss"]:.5f}')
            b.metric("Illustrative take profit", f'{r["take_profit"]:.5f}')
        else:
            st.info("The rules did not produce a strong enough signal. Waiting is preferable to forcing a trade.")
        d = st.session_state["analysis_data"]
        chart = go.Figure()
        x = d["timestamp"] if "timestamp" in d.columns else d.index
        chart.add_trace(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="EUR/USD"))
        chart.add_trace(go.Scatter(x=x, y=d["ema20"], name="EMA 20"))
        chart.add_trace(go.Scatter(x=x, y=d["ema50"], name="EMA 50"))
        chart.update_layout(height=520, xaxis_rangeslider_visible=False, title="EUR/USD candles and moving averages")
        st.plotly_chart(chart, use_container_width=True)
        st.download_button(
            "Download analysis report (CSV)",
            pd.DataFrame([{
                "signal": r["signal"], "score": r["score"], "price": r["price"],
                "RSI": r["rsi"], "EMA20": r["ema20"], "EMA50": r["ema50"],
                "BOS": r["bos"], "timeframe_bias": r["timeframe_bias"],
                "stop_loss": r["stop_loss"], "take_profit": r["take_profit"]
            }]).to_csv(index=False),
            file_name="eurusd_analysis_report.csv",
            mime="text/csv"
        )

with tab2:
    st.subheader("Analyse a trading-chart screenshot")
    st.write("This uses Gemini vision. Add your Gemini API key for this session only; it is not saved in this app.")
    chart_file = st.file_uploader("Upload chart screenshot", type=["png", "jpg", "jpeg"], key="chart")
    api_key = st.text_input("Gemini API key", type="password", help="Do not share your key with anyone.")
    context = st.text_area("Optional context", placeholder="Example: EUR/USD, 5-minute chart; identify trend, structure and possible setup.")
    if st.button("Analyse screenshot", type="primary", use_container_width=True):
        if chart_file is None:
            st.error("Please upload a chart screenshot.")
        elif not api_key.strip():
            st.error("Please enter your Gemini API key.")
        else:
            try:
                from google import genai
                from PIL import Image
                image = Image.open(chart_file).convert("RGB")
                client = genai.Client(api_key=api_key.strip())
                prompt = f"""
You are reviewing a user-supplied EUR/USD trading chart screenshot.
{context}

Describe only what is visibly supported by the image:
1. Visible market direction and candle structure.
2. Visible indicators and their readings, only if legible.
3. Potential support/resistance and break of structure, if identifiable.
4. A cautious potential BUY, SELL, or NO TRADE view.
5. What price action would confirm or invalidate the idea.
6. If the screenshot is unclear or evidence conflicts, say NO TRADE.

Do not invent prices, indicator values, or unseen data. Do not claim certainty.
Do not present this as financial advice.
"""
                with st.spinner("Reviewing chart screenshot..."):
                    result = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[prompt, image]
                    )
                st.markdown("### Screenshot analysis")
                st.write(result.text or "The model returned no text.")
            except Exception as exc:
                st.error(f"Screenshot analysis failed: {type(exc).__name__}: {exc}")
                st.caption("Check that the API key is valid and that the Gemini model is available to your account.")

st.divider()
st.caption("This website does not place trades, connect to Quotex, or guarantee outcomes. Always validate signals independently.")
