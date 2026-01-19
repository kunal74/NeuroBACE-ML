import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
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

# --- CUSTOM STYLING ---
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
    
    .header-container {{
        display: flex;
        align-items: center;
        gap: 25px;
        margin-bottom: 5px;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
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

# --- HEADER WITH CUSTOM LOGO ---
# To use your image, save it as 'logo.png' in the same folder as this script.
try:
    with open("logo.png", "rb") as f:
        data = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{data}" width="85">'
except:
    # Fallback to a placeholder if the file is missing
    logo_html = '<div style="width:85px; height:85px; background:#38bdf8; border-radius:50%;"></div>'

st.markdown(f"""
    <div class="header-container">
        {logo_html}
        <h1 style="margin: 0; font-size: 3rem; font-weight: 800; letter-spacing: -1px;">NeuroBACE-ML</h1>
    </div>
    <p style="margin-left: 110px; margin-top: -20px; font-weight: 500; font-style: italic; opacity: 0.8; font-size: 1.1rem;">
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
    - **Identification:** Local Serial Naming (C-n)
    """)
