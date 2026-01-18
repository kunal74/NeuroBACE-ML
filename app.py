import streamlit as st
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import plotly.express as px
import requests
from streamlit_lottie import st_lottie

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NeuroBACE-ML", page_icon="🧠", layout="wide")

# --- ADAPTIVE THEME LOGIC ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'

# Sidebar Toggle
st.sidebar.title("NeuroBACE-ML")
theme_choice = st.sidebar.radio("Appearance Mode", ["Dark", "Light"], horizontal=True)
st.session_state.theme = theme_choice

# Define Theme Variables
if st.session_state.theme == 'Dark':
    bg, text, card, accent = "#0f172a", "#f8fafc", "#1e293b", "#38bdf8"
    plotly_temp = "plotly_dark"
else:
    bg, text, card, accent = "#ffffff", "#1e293b", "#f1f5f9", "#2563eb"
    plotly_temp = "plotly_white"

# --- VISIBILITY CSS OVERHAUL ---
st.markdown(f"""
    <style>
    /* Global Background and Text Visibility */
    .stApp {{ background-color: {bg} !important; color: {text} !important; }}
    
    /* Ensure all sidebar text is visible */
    [data-testid="stSidebar"] {{ background-color: {bg} !important; border-right: 1px solid {accent}33; }}
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {{ color: {text} !important; }}
    
    /* Header and Subheader Visibility */
    h1, h2, h3, h4, .stMarkdown p, label, .stWidget label {{ color: {text} !important; }}
    
    /* Metric Card Fix (Removing Ugly White Blocks) */
    [data-testid="stMetric"] {{ 
        background-color: {card} !important; 
        border: 1px solid {accent}44 !important; 
        border-radius: 12px; 
        padding: 15px !important; 
    }}
    [data-testid="stMetricLabel"] p {{ color: {text}77 !important; }}
    [data-testid="stMetricValue"] div {{ color: {accent} !important; }}

    /* Tab Visibility */
    .stTabs [data-baseweb="tab"] {{ color: {text}99 !important; font-weight: 600; }}
    .stTabs [aria-selected="true"] {{ color: {accent} !important; border-bottom: 3px solid {accent} !important; }}

    /* Modern Button */
    .stButton>button {{ 
        background: linear-gradient(90deg, #0ea5e9, #2563eb); 
        color: white !important; border: none; font-weight: bold; border-radius: 8px; width: 100%;
    }}
    
    /* Hide broken icons */
    img {{ display: none !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- NEURAL MOTION GRAPHICS ---
@st.cache_data
def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

# High-contrast brain for both modes
brain_ani = load_lottie("https://lottie.host/8e2f8087-0b1a-4d76-9d8a-669e9c70c0c0/XlV5Z7Y4C6.json")

with st.sidebar:
    if brain_ani:
        st_lottie(brain_ani, height=180, key="main_ani")
    st.markdown("---")
    threshold = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.70, 0.01)
    st.caption("v1.0.2 | Professional Edition")

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
st.markdown("##### *High-Fidelity Virtual Screening for Alzheimer's Therapeutic Discovery*")
st.write("---")

t1, t2, t3 = st.tabs(["🚀 Screening Engine", "📈 Results Visualization", "🔬 Technical Specs"])

with t1:
    in_type = st.radio("Input Source", ["Manual Entry", "Batch Upload (CSV)"], horizontal=True)
    mols = []
    if in_type == "Manual Entry":
        raw = st.text_area("SMILES (one per line):", "COc1cc2c(cc1OC)C(=O)C(CC2)Cc3ccn(cc3)Cc4ccccc4")
        mols = [s.strip() for s in raw.split('\n') if s.strip()]
    else:
        f = st.file_uploader("Upload CSV (must contain 'smiles' column)")
        if f: 
            df = pd.read_csv(f)
            mols = df['smiles'].tolist() if 'smiles' in df.columns else []

    if st.button("Start Virtual Screening"):
        if model and mols:
            res = []
            for s in mols:
                p = run_prediction(s)
                if p is not None:
                    res.append({"SMILES": s, "Prob": p, "Result": "ACTIVE" if p >= threshold else "INACTIVE"})
            
            df_res = pd.DataFrame(res)
            st.session_state['results'] = df_res
            
            # Metrics (Now themed correctly)
            c1, c2, c3 = st.columns(3)
            c1.metric("Molecules", len(df_res))
            c2.metric("Potent Hits", len(df_res[df_res['Result'] == "ACTIVE"]))
            c3.metric("Max Prob", f"{df_res['Prob'].max():.2%}")
            
            st.dataframe(df_res.style.background_gradient(subset=['Prob'], cmap='Blues'), use_container_width=True)
            st.download_button("Export CSV", df_res.to_csv(index=False), "NeuroBACE_Report.csv")

with t2:
    if 'results' in st.session_state:
        fig = px.bar(st.session_state['results'], x='SMILES', y='Prob', color='Prob', 
                     color_continuous_scale='Blues', template=plotly_temp)
        st.plotly_chart(fig, use_container_width=True)

with t3:
    st.write("### Platform Specifications")
    st.markdown("""
    - **Architecture:** Optimized XGBoost Framework
    - **Performance:** Precision: 0.8695 | F1: 0.8801
    - **Dataset:** Human BACE1 (ChEMBL4822)
    """)
