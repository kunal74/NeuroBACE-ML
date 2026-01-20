import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import base64
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator # Modern API Import
import plotly.express as px
import os

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

# --- CUSTOM CSS ---
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
    st.caption("v1.2")

# --- PREDICTION ENGINE (JSON NATIVE) ---
@st.cache_resource
def load_model():
    json_file = 'BACE1_optimized_model.json'
    if not os.path.exists(json_file):
        st.error(f"⚠️ CRITICAL ERROR: File '{json_file}' not found.")
        return None
    try:
        model = xgb.Booster()
        model.load_model(json_file)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# --- FINGERPRINT GENERATOR (MODERN API) ---
# Initializing the modern generator to replace legacy AllChem calls
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def run_prediction(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Future-proof generation using GetFingerprintAsNumPy
            fp = mfpgen.GetFingerprintAsNumPy(mol)
            dmatrix = xgb.DMatrix(fp.reshape(1, -1))
            prediction = model.predict(dmatrix)
            return round(float(prediction[0]), 4)
    except Exception as e:
        return None
    return None

# --- HEADER ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
            return f'data:image/png;base64,{data}'
    except: return None

logo_url = get_base64_image("logo.png") 

if logo_url:
    logo_html = f'<img src="{logo_url}" width="120" style="margin-right: 15px;">'
else:
    logo_html = f'<div style="width:100px; height:100px; background:{accent}; border-radius:50%; margin-right: 20px;"></div>'

st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        {logo_html}
        <div>
            <h1 style="margin: 0; padding: 0; line-height: 1;">NeuroBACE-ML</h1>
            <p style="margin: 0; font-weight: 500; font-style: italic; opacity: 0.8; font-size: 1.1rem;">
                Advanced Predictive Framework for BACE1 Inhibition
            </p>
        </div>
    </div>
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
        if model is None:
            st.error("❌ Model not loaded.")
        elif not mols:
            st.warning("⚠️ No SMILES found.")
        else:
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
            
            if res:
                df_res = pd.DataFrame(res)
                st.session_state['results'] = df_res
                
                c1_m, c2_m, c3_m = st.columns(3)
                c1_m.metric("Molecules", len(df_res))
                c2_m.metric("Potent Hits", len(df_res[df_res['Result'] == "ACTIVE"]))
                c3_m.metric("Max Probability", f"{df_res['Inhibition Prob'].max():.2%}")
                
                st.write("---")
                st.dataframe(
                    df_res.style.background_gradient(subset=['Inhibition Prob'], cmap='RdYlGn'), 
                    use_container_width=True,
                    hide_index=True
                )
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

# --- TAB 3: SPECIFICATIONS (UPDATED) ---
with t3:
    st.write("### Platform Architecture")
    st.markdown("""
    - **Inference Engine:** XGBoost Framework (Native JSON Serialization)
    - **Molecular Encoding:** RDKit MorganGenerator (2048-bit, Radius=2) [Modern API]
    - **Optimization:** Bayesian Hyperparameter Tuning via Optuna
    - **Identification:** Local Serial Nomenclature (C-n)
    """)
