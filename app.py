import streamlit as st
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NeuroBACE-ML", page_icon="🧠", layout="wide")

# --- THEME LOGIC ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'

st.sidebar.title("NeuroBACE-ML")
theme_choice = st.sidebar.radio("Appearance Mode", ["Dark", "Light"], horizontal=True)
st.session_state.theme = theme_choice

if st.session_state.theme == 'Dark':
    bg, text, card, accent = "#0f172a", "#f8fafc", "#1e293b", "#38bdf8"
    plotly_temp = "plotly_dark"
else:
    bg, text, card, accent = "#ffffff", "#000000", "#f1f5f9", "#2563eb"
    plotly_temp = "plotly_white"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg} !important; color: {text} !important; }}
    [data-testid="stSidebar"] {{ background-color: {bg} !important; border-right: 1px solid {accent}33; }}
    h1, h2, h3, h4, label, span, p, [data-testid="stWidgetLabel"] p, .stMarkdown p {{ 
        color: {text} !important; opacity: 1 !important; 
    }}
    [data-testid="stMetric"] {{ background-color: {card} !important; border: 1px solid {accent}44 !important; border-radius: 12px; }}
    [data-testid="stMetricValue"] div {{ color: {accent} !important; font-weight: bold; }}
    .stButton>button {{ 
        background: linear-gradient(90deg, #0ea5e9, #2563eb) !important; 
        color: white !important; font-weight: bold !important; border-radius: 8px !important;
    }}
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("---")
    threshold = st.slider("Probability Threshold (P ≥ 0.7 = Active)", 0.0, 1.0, 0.70, 0.01)
    st.caption("v1.0")

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
# Professional Title with Inline SVG Logo
st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <svg width="45" height="45" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 15px;">
            <path d="M9.5 13C10.3284 13 11 12.3284 11 11.5C11 10.6716 10.3284 10 9.5 10C8.67157 10 8 10.6716 8 11.5C8 12.3284 8.67157 13 9.5 13Z" fill="{accent}"/>
            <path d="M14.5 13C15.3284 13 16 12.3284 16 11.5C16 10.6716 15.3284 10 14.5 10C13.6716 10 13 10.6716 13 11.5C13 12.3284 13.6716 13 14.5 13Z" fill="{accent}"/>
            <path d="M12 16C12.8284 16 13.5 15.3284 13.5 14.5C13.5 13.6716 12.8284 13 12 13C11.1716 13 10.5 13.6716 10.5 14.5C10.5 15.3284 11.1716 16 12 16Z" fill="{accent}"/>
            <path d="M12 9C12.8284 9 13.5 8.32843 13.5 7.5C13.5 6.67157 12.8284 6 12 6C11.1716 6 10.5 6.67157 10.5 7.5C10.5 8.32843 11.1716 9 12 9Z" fill="{accent}"/>
            <path d="M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="{accent}" stroke-width="1.5"/>
            <path d="M9.5 11.5L12 7.5L14.5 11.5L12 14.5L9.5 11.5Z" stroke="{accent}" stroke-width="1" stroke-dasharray="2 2"/>
        </svg>
        <h1 style="margin: 0; font-weight: 800; letter-spacing: -1px; color: {text};">NeuroBACE-ML</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("##### *Precision Platform for BACE1 Inhibitor Discovery*")
st.write("---")

# Using modern Material Icons for Tabs
t1, t2, t3 = st.tabs([":material/science: Screening Engine", ":material/monitoring: Visual Analytics", ":material/settings: Specifications"])

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
            bar = st.progress(0)
            for i, s in enumerate(mols):
                p = run_prediction(s)
                if p is not None:
                    res.append({
                        "Compounds": f"C-{i+1}", 
                        "Inhibition Prob": p, 
                        "Result": "ACTIVE" if p >= threshold else "INACTIVE",
                        "SMILES": s
                    })
                bar.progress((i + 1) / len(mols))
            
            df_res = pd.DataFrame(res)
            st.session_state['results'] = df_res
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Molecules", len(df_res))
            c2.metric("Potent Hits", len(df_res[df_res['Result'] == "ACTIVE"]))
            c3.metric("Max Probability", f"{df_res['Inhibition Prob'].max():.2%}")
            
            st.write("---")
            st.dataframe(df_res.style.background_gradient(subset=['Inhibition Prob'], cmap='RdYlGn'), use_container_width=True)
            st.download_button("Export Results", df_res.to_csv(index=False), "NeuroBACE_Report.csv")

with t2:
    if 'results' in st.session_state:
        st.markdown("### Predictive Probability Distribution")
        data = st.session_state['results'].sort_values('Inhibition Prob', ascending=True)
        
        fig = px.bar(
            data, 
            y='Compounds', 
            x='Inhibition Prob', 
            orientation='h',
            color='Inhibition Prob',
            color_continuous_scale=[[0, 'red'], [0.5, 'yellow'], [1, 'green']],
            template=plotly_temp,
            labels={'Inhibition Prob': 'Probability Score'},
            height=max(400, len(data) * 30)
        )
        
        fig.update_layout(xaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

with t3:
    st.write("### Platform Architecture")
    st.markdown("""
    - **Architecture:** XGBoost Framework (Pickle Serialization)
    - **Optimization:** Bayesian Framework via Optuna
    - **Feature Extraction:** 2048-bit Morgan Fingerprints (Radius=2)
    - **Identification:** Local Serial Naming (C-n)
    """)
