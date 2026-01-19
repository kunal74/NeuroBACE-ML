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

# --- CUSTOM CSS FOR ALIGNMENT & VISIBILITY ---
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
    
    /* Header Container for Perfect Alignment */
    .header-container {{
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 5px;
    }}
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

# --- MAIN DASHBOARD HEADER ---
# This block fixes the "question mark" and "misalignment" issues simultaneously
st.markdown(f"""
    <div class="header-container">
        <svg width="70" height="70" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13.03 19.89C12.7 19.96 12.36 20 12 20C10.28 20 8.68 19.46 7.36 18.54C8.57 17.2 10.21 16.31 12.06 16.04C13.91 15.77 15.82 15.97 17.5 16.61C16.4 18.35 14.87 19.6 13.03 19.89ZM18.93 15.1C17.06 14.26 14.96 14 12.91 14.29C10.86 14.58 8.99 15.53 7.6 17.02C6.24 15.67 5.32 13.89 5.07 11.93C6.54 11.3 8.14 11 9.8 11C11.38 11 12.92 11.27 14.37 11.82C15.82 12.37 17.13 13.23 18.17 14.36C18.48 14.09 18.74 14.35 18.93 15.1ZM19.41 12.69C18.26 11.47 16.84 10.54 15.28 9.95C13.72 9.36 12.07 9.05 10.38 9.05C8.55 9.05 6.8 9.38 5.21 10.05C5.06 9.7 5.01 9.35 5.01 9C5.01 5.13 8.13 2 12 2C15.87 2 19 5.13 19 9C19 10.42 18.58 11.73 17.86 12.84C18.46 12.7 18.98 12.67 19.41 12.69ZM12 4C9.24 4 7 6.24 7 9C7 9.17 7.01 9.34 7.02 9.51C8.11 9.19 9.24 9.02 10.38 9.02C12.21 9.02 13.98 9.37 15.65 10.04C16.47 9.86 17 9.13 17 8.25C17 6.24 14.76 4 12 4Z" fill="{accent}"/>
        </svg>
        <h1 style="margin: 0; padding: 0;">NeuroBACE-ML</h1>
    </div>
    <p style="margin-left: 90px; margin-top: -15px; font-weight: 500; font-style: italic; opacity: 0.8;">
        Precision Platform for BACE1 Inhibitor Discovery
    </p>
""", unsafe_allow_html=True)

st.write("---")

t1, t2, t3 = st.tabs([":material/science: Screening Engine", ":material/monitoring: Visual Analytics", ":material/settings: Specifications"])

# --- TAB 1: SCREENING ENGINE ---
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
            
            c1_m, c2_m, c3_m = st.columns(3)
            c1_m.metric("Molecules", len(df_res))
            c2_m.metric("Potent Hits", len(df_res[df_res['Result'] == "ACTIVE"]))
            c3_m.metric("Max Probability", f"{df_res['Inhibition Prob'].max():.2%}")
            
            st.write("---")
            st.dataframe(df_res.style.background_gradient(subset=['Inhibition Prob'], cmap='RdYlGn'), use_container_width=True)
            st.download_button("Export Results", df_res.to_csv(index=False), "NeuroBACE_Report.csv")

# --- TAB 2: VISUAL ANALYTICS ---
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

# --- TAB 3: SPECIFICATIONS ---
with t3:
    st.write("### Platform Architecture")
    st.markdown("""
    - **Architecture:** XGBoost Framework (Pickle Serialization)
    - **Optimization:** Bayesian Framework via Optuna
    - **Feature Extraction:** 2048-bit Morgan Fingerprints (Radius=2)
    - **Nomenclature:** Internal Serial Naming (C-n)
    """)
