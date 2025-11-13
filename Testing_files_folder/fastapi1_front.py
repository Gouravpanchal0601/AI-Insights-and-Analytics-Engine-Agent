import streamlit as st
import pandas as pd
import requests
import json
import plotly.io as pio
import plotly.graph_objects as go

# =======================
# CONFIG
# =======================
FASTAPI_URL = "http://127.0.0.1:8000"  # backend base URL

st.set_page_config(page_title="🩺 AI Insights & Analytics Dashboard", layout="wide")

st.title("🩺 AI Insights & Analytics Engine")
st.markdown("Upload your dataset and interact with the FastAPI backend for analysis.")

# =======================
# UPLOAD CSV
# =======================
uploaded_file = st.file_uploader("📂 Upload your CSV", type=["csv"])

if uploaded_file:
    with st.spinner("Uploading to backend..."):
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
        resp = requests.post(f"{FASTAPI_URL}/upload", files=files)
    if resp.status_code == 200:
        st.success("✅ File uploaded successfully to FastAPI!")
        info = resp.json()
        st.json(info)
    else:
        st.error(f"Upload failed: {resp.text}")

# =======================
# SHOW DATA SUMMARY
# =======================
if st.button("📊 Generate Summary"):
    resp = requests.get(f"{FASTAPI_URL}/summary")
    if resp.status_code == 200:
        summary = resp.json().get("summary", "")
        st.markdown(summary)
    else:
        st.error("❌ Failed to get summary.")

# =======================
# VISUALIZATIONS
# =======================
st.subheader("📈 Visualizations")

col1, col2 = st.columns(2)

with col1:
    column = st.text_input("Column for Histogram", "age")
    if st.button("Show Histogram"):
        resp = requests.get(f"{FASTAPI_URL}/visualize/histogram", params={"column": column})
        if resp.status_code == 200:
            fig = pio.from_json(json.dumps(resp.json()))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ Could not generate histogram.")

with col2:
    x = st.text_input("X-axis", "age")
    y = st.text_input("Y-axis", "total_bill")
    if st.button("Show Scatter Plot"):
        resp = requests.get(f"{FASTAPI_URL}/visualize/scatter", params={"x": x, "y": y})
        if resp.status_code == 200:
            fig = pio.from_json(json.dumps(resp.json()))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ Could not generate scatter plot.")

if st.button("🔍 Show Correlation Heatmap"):
    resp = requests.get(f"{FASTAPI_URL}/visualize/heatmap")
    if resp.status_code == 200 and "data" in resp.json():
        fig = pio.from_json(json.dumps(resp.json()))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Could not generate heatmap.")

# =======================
# CHAT WITH DATA
# =======================
st.markdown("---")
st.subheader("💬 Chat with Your Data")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask something (e.g. 'Show all patients above 60')", key="chat_input")

if st.button("🚀 Send Question"):
    if user_input.strip():
        st.session_state.chat_history.append(("You", user_input))
        payload = {"question": user_input}
        resp = requests.post(f"{FASTAPI_URL}/chat", json=payload)
        if resp.status_code == 200:
            answer = resp.json().get("response", "")
            st.session_state.chat_history.append(("AI", answer))
        else:
            st.session_state.chat_history.append(("AI", "⚠️ Backend error."))

# Display chat history
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**👤 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")

if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
