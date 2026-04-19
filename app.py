import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Attention-Aware Summarizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS — Clean, Light, Professional
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg:        #F7F6F2;
    --surface:   #FFFFFF;
    --border:    #E2DDD6;
    --text:      #1A1714;
    --muted:     #7A756D;
    --accent:    #C84B31;
    --accent2:   #2D6A4F;
    --accent3:   #1D3557;
    --radius:    10px;
    --shadow:    0 2px 12px rgba(0,0,0,0.07);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    color: var(--text);
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 4px;
    border: 1px solid var(--border);
    gap: 4px;
}
[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    border-radius: 7px !important;
    transition: all .2s;
}
[aria-selected="true"] {
    background: var(--accent) !important;
    color: #fff !important;
}

/* ── Text area ── */
textarea {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: var(--radius);
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    letter-spacing: .03em;
    padding: .55rem 1.5rem;
    transition: opacity .2s, transform .15s;
}
[data-testid="stButton"] > button:hover {
    opacity: .88;
    transform: translateY(-1px);
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
}
.summary-text {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    line-height: 1.75;
    color: var(--text);
}
.metric-chip {
    display: inline-block;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: .78rem;
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    margin-right: 6px;
}
.section-label {
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .4rem;
}
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 4px;
    font-size: .72rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}
.badge-green  { background: #d8f3dc; color: #2D6A4F; }
.badge-red    { background: #fde8e3; color: #C84B31; }
.badge-blue   { background: #dde8f5; color: #1D3557; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# API Configuration
# ─────────────────────────────────────────────
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sshleifer/distilbart-cnn-12-6"

def generate_summary_api(text, token):
    """Call the model for  summarization."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 130,
            "min_length": 40,
            "do_sample": False
        }
    }
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.text}")
    
    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("summary_text", "")
    return ""

# ─────────────────────────────────────────────
# Visualization Helpers (Placeholders for API)
# ─────────────────────────────────────────────
def clean_tokens(tokens):
    return [t.replace("Ġ", " ").replace("Ċ", "↵").strip() or "␣" for t in tokens]

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    hf_token = st.secrets["HF_TOKEN"]

    st.divider()
    st.markdown("## ⚙️ Settings")
    st.caption("Using Model for inference.")

    st.divider()
    
    

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.8rem 0 0.8rem">
  <p class="section-label"> Model Text Summarizer</p>
  <h1 style="margin:0; font-size:2.2rem; line-height:1.1">
    <br>Text Summarizer
  </h1>
  <p style="color:#7A756D; margin-top:.5rem; font-size:.95rem; max-width:520px">
    Summarize long text instantly using the <em>encoder decoder model</em> model.
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────
DEMO_TEXT = (
    "Scientists at MIT have developed a new type of solar cell that can generate "
    "electricity even on cloudy days. The technology uses a perovskite-silicon tandem "
    "structure to capture a broader spectrum of sunlight. In lab tests, the cells achieved "
    "a record efficiency of 33.2%, surpassing the theoretical limit of conventional silicon "
    "cells. The researchers believe the technology could be commercially available within "
    "five years. Climate experts say this breakthrough could significantly accelerate the "
    "global transition to renewable energy and reduce dependence on fossil fuels."
)

input_text = st.text_area(
    "Source text",
    value=DEMO_TEXT,
    height=160,
    placeholder="Paste a paragraph or article excerpt here…",
)

col_btn, col_stat = st.columns([1, 3])
with col_btn:
    run = st.button("✦ Summarize", use_container_width=True)

# ─────────────────────────────────────────────
# Main Logic
# ─────────────────────────────────────────────
if run:
            
    if not input_text.strip():
        st.warning("Please enter some text first.")
        st.stop()

    with st.spinner(" "):
        try:
            summary = generate_summary_api(input_text, hf_token)
            st.session_state["summary"] = summary
        except Exception as e:
            st.error(f"Error : {e}")
            st.stop()

# ─────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────
if "summary" in st.session_state:
    summary = st.session_state["summary"]

    # ── Summary Card ──
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Generated Summary</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="summary-text">{summary}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Stats</p>', unsafe_allow_html=True)
        orig_words  = len(input_text.split())
        summ_words  = len(summary.split())
        ratio       = round(summ_words / orig_words * 100) if orig_words else 0
        st.metric("Original words", orig_words)
        st.metric("Summary words",  summ_words)
        st.metric("Compression",    f"{ratio}%")
        st.markdown(
            f'<span class="badge badge-green">API CLOUD</span>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    
else:
    st.info("Enter text and click **✦ Summarize** to begin.")
