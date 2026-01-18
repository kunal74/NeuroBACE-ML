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

# --- THEME TOGGLE LOGIC ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'

theme_choice = st.sidebar.radio("Appearance Mode", ["Dark", "Light"], horizontal=True)
st.session_state.theme = theme_choice

# --- DYNAMIC CSS FOR THEME SWITCHING ---
if st.session_state.theme == 'Dark':
    bg_color, text_color, card_bg, accent = "#0f172a", "#f8fafc", "#1e293b", "#38bdf8"
    plotly_template = "plotly_dark"
else:
    bg_color, text_color, card_bg, accent = "#ffffff", "#0f172a", "#f1f5f9", "#2563eb"
    plotly_template = "plotly_white"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    [data-testid="stSidebar"] {{ background-color: {bg_color} !important; border-right: 1px solid {accent}33; }}
    h1, h2, h3, h4 {{ color: {accent} !important; }}
    [data-testid="stMetric"] {{ background-color: {card_bg} !important; border-radius: 12px; border: 1px solid {accent}22; }}
    .stButton>button {{ background: linear-gradient(90deg, #0ea5e9, #2563eb); color: white; border: none; font-weight: bold; border-radius: 8px; }}
    .stTabs [data-baseweb="tab-list"] {{ background-color: transparent; }}
    .stTabs [data-baseweb="tab"] {{ color: {text_color}77; }}
    .stTabs [aria-selected="true"] {{ color: {accent} !important; border-bottom: 2px solid {accent} !important; }}
    img {{ display: none !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- LOAD LOTTIE ANIMATION (High Contrast Neural Network) ---
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

# High-contrast neural brain animation
lottie_brain = load_lottieurl("https://lottie.host/8e2f8087-0b1a-4d76-9d8a-669e9c70c0c0/XlV5Z7Y4C6.json")

# --- SIDEBAR: NAVIGATION ---
with st.sidebar:
    if lottie_brain:
        st_lottie(lottie_brain, height=180, key="brain_main")
    st.title("NeuroBACE-ML")
    st.markdown("---")
    threshold = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.70, 0.01)
    st.caption("v1.0.1 | Adaptive Interface")

# --- CORE LOGIC ---
@st.cache_resource
def load_neurobace_model():
    try:
        with open('BACE1_trained_model_optimized.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

model = load_neurobace_model()

def predict_compound(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_array = np.array(fp).reshape(1, -1)
        return round(model.predict_proba(fp_array)[0][1], 4)
    return None

# --- MAIN DASHBOARD ---
st.title("🧠 NeuroBACE-ML")
st.markdown("#### *High-Fidelity Virtual Screening for Alzheimer's Therapeutic Discovery*")
st.write("---")

tab1, tab2, tab3 = st.tabs(["🚀 Screening Engine", "📈 Results Visualization", "🔬 Technical Specs"])

with tab1:
    input_type = st.radio("Input Source", ["Manual Entry", "Batch Upload (CSV)"], horizontal=True)
    compounds = []
    if input_type == "Manual Entry":
        raw = st.text_area("SMILES (one per line):", "COc1cc2c(cc1OC)C(=O)C(CC2)Cc3ccn(cc3)Cc4ccccc4")
        compounds = [s.strip() for s in raw.split('\n') if s.strip()]
    else:
        f = st.file_uploader("Upload CSV (must contain 'smiles' column)"); 
        if f: df = pd.read_csv(f); compounds = df['smiles'].tolist() if 'smiles' in df.columns else []

    if st.button("Start Virtual Screening"):
        if model and compounds:
            results = []
            for s in compounds:
                p = predict_compound(s)
                if p is not None:
                    results.append({"SMILES": s, "Inhibition_Prob": p, "Result": "ACTIVE" if p >= threshold else "INACTIVE"})
            
            res_df = pd.DataFrame(results); st.session_state['results'] = res_df
            c1, c2, c3 = st.columns(3)
            c1.metric("Molecules", len(res_df))
            c2.metric("Predicted Actives", len(res_df[res_df['Result'] == "ACTIVE"]))
            c3.metric("Max Probability", f"{res_df['Inhibition_Prob'].max():.2%}")
            
            st.dataframe(res_df.style.background_gradient(subset=['Inhibition_Prob'], cmap='Blues'), use_container_width=True)
            st.download_button("Export CSV", res_df.to_csv(index=False), "NeuroBACE_ML_Report.csv")

with tab2:
    if 'results' in st.session_state:
        fig = px.bar(st.session_state['results'], x='SMILES', y='Inhibition_Prob', color='Inhibition_Prob', color_continuous_scale='Blues', template=plotly_template)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.write("### Platform Data")
    st.markdown("- **Precision Score:** 0.8695 | **F1-Score:** 0.8801\n- **Optimized Framework:** XGBoost + Optuna")
