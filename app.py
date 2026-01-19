import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
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

# Adaptive Colors for High Contrast
if st.session_state.theme == 'Dark':
    bg, text, card, accent = "#0f172a", "#f8fafc", "#1e293b", "#38bdf8"
    plotly_temp = "plotly_dark"
else:
    # High-contrast Light Theme
    bg, text, card, accent = "#ffffff", "#000000", "#f1f5f9", "#2563eb"
    plotly_temp = "plotly_white"

# --- UNIVERSAL VISIBILITY CSS ---
st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg} !important; color: {text} !important; }}
    [data-testid="stSidebar"] {{ background-color: {bg} !important; border-right: 1px solid {accent}33; }}
    
    /* Force text visibility */
    h1, h2, h3, h4, label, span, p, [data-testid="stWidgetLabel"] p, .stMarkdown p {{ 
        color: {text} !important; opacity: 1 !important; 
    }}
    
    [data-testid="stMetric"] {{ background-color: {card} !important; border: 1px solid {accent}44 !important; border-radius: 12px; }}
    [data-testid="stMetricValue"] div {{ color: {accent} !important; font-weight: bold; }}

    /* Button Styling */
    .stButton>button {{ 
        background: linear-gradient(90deg, #0ea5e9, #2563eb) !important; 
        color: white !important; font-weight: bold !important; border-radius: 8px !important;
    }}
    
    img {{ display: none !important; }}
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

# --- UTILITY: NAME RECOGNITION ---
from urllib.parse import quote  # Add this at the top of your app.py

def get_compound_name(smiles):
    try:
        # 1. URL-encode the SMILES string (fixes characters like #, =, and ())
        encoded_smi = quote(smiles)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smi}/property/Title/JSON"
        
        # 2. Add a 'verify=False' if you are on a restricted network (use with caution)
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            return response.json()['PropertyTable']['Properties'][0].get('Title', "Unknown")
        else:
            return f"Not Found (Code: {response.status_code})"
    except Exception as e:
        # This will tell you the ACTUAL error (e.g., Timeout, Connection Refused)
        return f"Conn Error: {type(e).__name__}"

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("---")
    threshold = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.70, 0.01)
    st.caption("v1.3.0 | Traffic Light Color Scheme")

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
            bar = st.progress(0)
            for i, s in enumerate(mols):
                p = run_prediction(s)
                if p is not None:
                    res.append({
                        "Compound Name": get_compound_name(s),
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
            # Apply Red-Yellow-Green (RdYlGn) gradient to the table
            st.dataframe(df_res.style.background_gradient(subset=['Inhibition Prob'], cmap='RdYlGn'), use_container_width=True)
            st.download_button("Export Results", df_res.to_csv(index=False), "NeuroBACE_Report.csv")

with t2:
    if 'results' in st.session_state:
        st.markdown("### Predictive Probability Distribution")
        # Sort data for a cleaner visual gradient
        data = st.session_state['results'].sort_values('Inhibition Prob', ascending=True)
        
        # Horizontal bar chart with matched Red-Yellow-Green scale
        fig = px.bar(
            data, 
            y='Compound Name', 
            x='Inhibition Prob', 
            orientation='h',
            color='Inhibition Prob',
            # Strict mapping: Red(0) -> Yellow(0.5) -> Green(1)
            color_continuous_scale=[[0, 'red'], [0.5, 'yellow'], [1, 'green']],
            template=plotly_temp,
            labels={'Inhibition Prob': 'Probability Score'},
            height=max(400, len(data) * 30) # Prevent overlapping names
        )
        
        fig.update_layout(xaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

with t3:
    st.write("### Platform Architecture")
    st.markdown("""
    - **Architecture:** Optimized XGBoost Framework
    - **Optimization:** Bayesian Framework via Optuna
    - **Feature Extraction:** 2048-bit Morgan Fingerprints (Radius=2)
    - **Database Integration:** PubChem PUG REST API for real-time name recognition
    """)
