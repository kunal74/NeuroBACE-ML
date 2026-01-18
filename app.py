import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from rdkit import Chem
from rdkit.Chem import AllChem
import plotly.express as px
from streamlit_lottie import st_lottie

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NeuroBACE-ML", page_icon="🧠", layout="wide")

# --- THEME STATE ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'

st.sidebar.title("NeuroBACE-ML")
theme_choice = st.sidebar.radio("Appearance Mode", ["Dark", "Light"], horizontal=True)
st.session_state.theme = theme_choice

# Adaptive Colors for Perfect Visibility
if st.session_state.theme == 'Dark':
    bg, text, card, accent = "#0f172a", "#f8fafc", "#1e293b", "#38bdf8"
    plotly_temp = "plotly_dark"
else:
    bg, text, card, accent = "#ffffff", "#000000", "#f1f5f9", "#2563eb"
    plotly_temp = "plotly_white"

# --- UNIVERSAL VISIBILITY CSS ---
st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg} !important; color: {text} !important; }}
    [data-testid="stSidebar"] {{ background-color: {bg} !important; border-right: 1px solid {accent}33; }}
    h1, h2, h3, h4, label, span, p, [data-testid="stWidgetLabel"] p, .stMarkdown p {{ 
        color: {text} !important; opacity: 1 !important; 
    }}
    [data-testid="stMetric"] {{ background-color: {card} !important; border: 1px solid {accent}44 !important; border-radius: 12px; }}
    [data-testid="stMetricValue"] div {{ color: {accent} !important; font-weight: bold; }}
    .stButton>button {{ background: linear-gradient(90deg, #0ea5e9, #2563eb) !important; color: white !important; font-weight: bold !important; border-radius: 8px !important; }}
    img {{ display: none !important; }}
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

# --- LOCAL NEURAL MOTION GRAPHICS ---
def load_lottie_local(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except: return None

brain_ani = load_lottie_local("brain_animation.json")

with st.sidebar:
    if brain_ani:
        st_lottie(brain_ani, height=200, key="main_ani", speed=1)
    else:
        st.error("⚠️ Upload 'brain_animation.json' to GitHub to see graphics.")
    st.markdown("---")
    threshold = st.sidebar.slider("Sensitivity Threshold", 0.0, 1.0, 0.70, 0.01)

# --- PREDICTION ENGINE ---
@st.cache_resource
def load_model():
    try:
        with open('BACE1_trained_model_optimized.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

model = load_model()

def run_prediction(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return round(model.predict_proba(np.array(fp).reshape(1, -1))[0][1], 4)
    return None

# --- MAIN DASHBOARD ---
st.title("🧠 NeuroBACE-ML")
st.markdown("##### *Advanced Platform for BACE1 Inhibitor Prediction*")
st.write("---")

t1, t2, t3 = st.tabs(["🚀 Screening Engine", "📈 Visual Analytics", "🔬 Specifications"])

with t1:
    in_type = st.radio("Input Source", ["Manual Entry", "Batch Upload (CSV)"], horizontal=True)
    mols = []
    if in_type == "Manual Entry":
        raw = st.text_area("SMILES (one per line):", "COc1cc2c(cc1OC)C(=O)C(CC2)Cc3ccn(cc3)Cc4ccccc4")
        mols = [s.strip() for s in raw.split('\n') if s.strip()]
    else:
        f = st.file_uploader("Upload CSV")
        if f: 
            df_in = pd.read_csv(f)
            mols = df_in['smiles'].tolist() if 'smiles' in df_in.columns else []

    if st.button("Start Virtual Screening"):
        if model and mols:
            res = []
            for s in mols:
                p = run_prediction(s)
                if p is not None:
                    res.append({"SMILES": s, "Prob": p, "Result": "ACTIVE" if p >= threshold else "INACTIVE"})
            df_res = pd.DataFrame(res)
            st.session_state['results'] = df_res
            c1, c2, c3 = st.columns(3)
            c1.metric("Molecules", len(df_res))
            c2.metric("Potent Hits", len(df_res[df_res['Result'] == "ACTIVE"]))
            c3.metric("Max Prob", f"{df_res['Prob'].max():.2%}")
            st.write("---")
            st.dataframe(df_res.style.background_gradient(subset=['Prob'], cmap='Blues'), use_container_width=True)
            st.download_button("Export Results", df_res.to_csv(index=False), "NeuroBACE_Report.csv")

with t2:
    if 'results' in st.session_state:
        fig = px.bar(st.session_state['results'], x='SMILES', y='Prob', color='Prob', 
                     color_continuous_scale='Blues', template=plotly_temp)
        st.plotly_chart(fig, use_container_width=True)

with t3:
    st.write("### Platform Specifications")
    st.markdown(f"""
    - **Balanced Accuracy**: 0.8619
    - **F1 Score**: 0.8801
    - **Precision**: 0.8695
    """)
