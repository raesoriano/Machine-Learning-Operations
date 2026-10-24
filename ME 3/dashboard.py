import streamlit as st
import pandas as pd
import time
import random
from datetime import datetime

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

# NOTE: All metrics on this dashboard are SIMULATED (random values).
# To show real telemetry, wire this up to read from the orchestrator's
# latency log / a socket / a shared file. See latency_benchmark.py for the
# categories that should be measured.
st.warning("⚠️ SIMULATED DATA — this is a UI mockup. Connect it to real telemetry to go live.")

# --- SIDEBAR: System Status ---
st.sidebar.header("System Status")
asr_status = st.sidebar.toggle("ASR Server", value=True)
tts_status = st.sidebar.toggle("TTS Server", value=True)
vad_status = st.sidebar.toggle("VAD System", value=True)

status_color = "🟢" if (asr_status and tts_status and vad_status) else "🔴"
st.sidebar.markdown(f"### Overall Health: {status_color}")

# --- TOP ROW: Real-time Latency Metrics ---
st.subheader("⚡ Live Latency Metrics (ms)")
col1, col2, col3, col4, col5 = st.columns(5)

# Simulated live data (In a real scenario, this would read from a log file or socket)
def get_simulated_latency(base, var):
    return base + random.randint(-var, var)

with col1:
    st.metric("VAD Trigger", f"{get_simulated_latency(40, 10)}ms", "-2ms")
with col2:
    st.metric("ASR Turnaround", f"{get_simulated_latency(450, 50)}ms", "12ms")
with col3:
    st.metric("LLM TTFB", f"{get_simulated_latency(800, 200)}ms", "-40ms")
with col4:
    st.metric("TTS Synthesis", f"{get_simulated_latency(300, 100)}ms", "5ms")
with col5:
    st.metric("Interruption", f"{get_simulated_latency(120, 30)}ms", "-10ms")

# --- MIDDLE ROW: Interaction Log ---
st.subheader("💬 Interaction Stream")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Simulate incoming data
if random.random() > 0.7:
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append({
        "Time": timestamp, 
        "Role": random.choice(["User", "OnIt"]), 
        "Text": random.choice(["Hello!", "How is the latency?", "Stop talking!", "Processing request..."]),
        "Latency": f"{random.randint(200, 1500)}ms"
    })

df_history = pd.DataFrame(st.session_state.chat_history).tail(10)
st.table(df_history)

# --- BOTTOM ROW: Performance Trends ---
st.subheader("📈 Performance Trends")
chart_data = pd.DataFrame({
    "VAD": [get_simulated_latency(40, 5) for _ in range(20)],
    "ASR": [get_simulated_latency(450, 50) for _ in range(20)],
    "LLM": [get_simulated_latency(800, 200) for _ in range(20)],
    "TTS": [get_simulated_latency(300, 100) for _ in range(20)],
})
st.line_chart(chart_data)

# --- FOOTER ---
st.markdown("---")
st.caption("Deployed on Raspberry Pi | Orchestrated by OnIt | Framework: Streamlit")
