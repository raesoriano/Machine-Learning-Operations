import streamlit as st
import pandas as pd
import random
import time
from datetime import datetime

from telemetry import default as telemetry, CATEGORIES

# Page config
st.set_page_config(
    page_title="OnIt Voice Agent Dashboard",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS for a "Dark Mode" AI feel
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e222d;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4451;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎙️ OnIt Full-Duplex Agent Monitor")
st.markdown("Real-time telemetry for the Raspberry Pi Voice Orchestrator")

# ------------------------------------------------------------------ #
# Telemetry source
# ------------------------------------------------------------------ #
# The dashboard reads the orchestrator's JSONL log (cross-process). If no
# records exist yet (agent not started, or no turns completed), it falls back
# to clearly-labelled simulated data so the UI is never blank.
records = telemetry.read_recent(max_lines=1000)
latency_records = [r for r in records if r.get("type") == "latency"]
interaction_records = [r for r in records if r.get("type") == "interaction"]

LIVE = len(latency_records) > 0

# --- SIDEBAR: System Status ---
st.sidebar.header("System Status")
st.sidebar.caption(f"Telemetry log: `{telemetry.log_file}`")
st.sidebar.caption(f"Records read: {len(records)} "
                   f"({len(latency_records)} latency, "
                   f"{len(interaction_records)} interaction)")
asr_status = st.sidebar.toggle("ASR Server", value=True)
tts_status = st.sidebar.toggle("TTS Server", value=True)
vad_status = st.sidebar.toggle("VAD System", value=True)

status_color = "🟢" if (asr_status and tts_status and vad_status) else "🔴"
st.sidebar.markdown(f"### Overall Health: {status_color}")

if LIVE:
    st.success("✅ LIVE — reading real telemetry from the orchestrator.")
else:
    st.warning("⚠️ No real telemetry yet — showing SIMULATED data. "
               "Start `agent_orchestrator.py` and complete a turn to go live.")

# Auto-refresh so the dashboard tracks the running agent.
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh now"):
    st.rerun()
st.sidebar.caption("Auto-refreshes every 5s while the page is open.")


def _cat_samples(cat):
    """Latency samples (ms) for a category from the real log, oldest first."""
    return [r["value_ms"] for r in latency_records
            if r.get("category") == cat]


def _latest(cat):
    samples = _cat_samples(cat)
    return samples[-1] if samples else None


def _delta(cat):
    samples = _cat_samples(cat)
    if len(samples) < 2:
        return None
    return samples[-1] - samples[-2]


def _sim(base, var):
    return base + random.randint(-var, var)


# --- TOP ROW: Real-time Latency Metrics ---
st.subheader("⚡ Live Latency Metrics (ms)")
col1, col2, col3, col4, col5 = st.columns(5)

# (column, label, category, simulated base/var)
metric_cols = [
    (col1, "VAD Trigger", "vad_trigger", 40, 10),
    (col2, "ASR Turnaround", "asr_turnaround", 450, 50),
    (col3, "LLM TTFB", "llm_ttfb", 800, 200),
    (col4, "TTS Synthesis", "tts_synthesis", 300, 100),
    (col5, "Interruption", "interruption", 120, 30),
]

for col, label, cat, base, var in metric_cols:
    with col:
        if LIVE:
            latest = _latest(cat)
            delta = _delta(cat)
            if latest is None:
                st.metric(label, "—", "no data yet")
            else:
                delta_str = (f"{delta:+.0f}ms"
                             if delta is not None else "first sample")
                st.metric(label, f"{latest:.0f}ms", delta_str)
        else:
            st.metric(label, f"{_sim(base, var)}ms", "simulated")

# --- MIDDLE ROW: Interaction Log ---
st.subheader("💬 Interaction Stream")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if LIVE:
    # Real interaction records from the log (newest last).
    real_history = [{
        "Time": datetime.fromtimestamp(r["ts"]).strftime("%H:%M:%S"),
        "Role": r.get("role", "?"),
        "Text": r.get("text", ""),
        "Latency": (f"{r['latency_ms']:.0f}ms"
                    if r.get("latency_ms") is not None else "—"),
    } for r in interaction_records]
    st.session_state.chat_history = real_history
else:
    # Simulated fallback.
    if random.random() > 0.7:
        st.session_state.chat_history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Role": random.choice(["User", "OnIt"]),
            "Text": random.choice(["Hello!", "How is the latency?",
                                   "Stop talking!", "Processing request..."]),
            "Latency": f"{random.randint(200, 1500)}ms",
        })

df_history = pd.DataFrame(st.session_state.chat_history).tail(10)
if df_history.empty:
    st.info("No interactions yet.")
else:
    st.table(df_history)

# --- BOTTOM ROW: Performance Trends ---
st.subheader("📈 Performance Trends")
if LIVE:
    # Build a trend series from the real samples (pad to a fixed width so the
    # chart is stable as data grows).
    width = 20
    def _trend(cat):
        samples = _cat_samples(cat)
        if not samples:
            return [None] * width
        # Take the most recent `width` samples, left-pad with NaN.
        recent = samples[-width:]
        return [None] * (width - len(recent)) + recent

    chart_data = pd.DataFrame({
        "VAD": _trend("vad_trigger"),
        "ASR": _trend("asr_turnaround"),
        "LLM": _trend("llm_ttfb"),
        "TTS": _trend("tts_synthesis"),
    })
    st.line_chart(chart_data)
else:
    chart_data = pd.DataFrame({
        "VAD": [_sim(40, 5) for _ in range(20)],
        "ASR": [_sim(450, 50) for _ in range(20)],
        "LLM": [_sim(800, 200) for _ in range(20)],
        "TTS": [_sim(300, 100) for _ in range(20)],
    })
    st.line_chart(chart_data)

# --- FOOTER ---
st.markdown("---")
st.caption("Deployed on Raspberry Pi | Orchestrated by OnIt | Framework: Streamlit")

# Auto-refresh: re-run the script every 5 seconds while the page is open.
time.sleep(5)
st.rerun()
