import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import base64
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import plotly.express as px
import os
import time

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
    threshold = st.slider("Decision Threshold (Validation Optimized)", 0.0, 1.0, 0.70, 0.01)
    st.caption("v1.2 | Robust Safety & Batching")

# --- PREDICTION ENGINE ---
@st.cache_resource
def load_model():
    json_file = 'BACE1_optimized_model.json'
    if not os.path.exists(json_file):
        st.error(f"🚨 Critical Error: Model file '{json_file}' is missing.")
        return None
    try:
        model = xgb.Booster()
        model.load_model(json_file)
        return model
    except xgb.core.XGBoostError as e:
        st.error(f"🚨 Model Error: {e}")
        return None
    except Exception as e:
        st.error(f"🚨 Unexpected Error: {e}")
        return None

model = load_model()
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# --- SCIENTIFIC HELPER FUNCTIONS ---
def get_confidence_level(prob):
    dist = abs(prob - 0.5)
    if dist < 0.15: 
        return "LOW (Ambiguous)"
    elif dist < 0.35: 
        return "MEDIUM"
    else:
        return "HIGH"

def get_fingerprint(smiles):
    """Generates explicit bit-vector fingerprint securely."""
    try:
        if not smiles or not isinstance(smiles, str):
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp_bitvect = mfpgen.GetFingerprint(mol)
            fp = np.zeros((2048,), dtype=np.int8) 
            DataStructs.ConvertToNumpyArray(fp_bitvect, fp)
            return fp
    except:
        return None
    return None

# FIX FOR DEFICIENCY #3: Robust Batch Inference Function
def run_batch_inference(model, valid_fps):
    """
    Defensively handles inference. 
    Returns raw probabilities or None if model/data is invalid.
    """
    # 1. Safety Guard: Explicitly check if model is None inside the function
    if model is None:
        return None 
    
    # 2. Safety Guard: Check for empty data
    if not valid_fps:
        return None

    try:
        # 3. Vectorized Prediction
        X_batch = np.vstack(valid_fps)
        dmatrix_batch = xgb.DMatrix(X_batch)
        return model.predict(dmatrix_batch)
    except Exception:
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
        f = st.file_uploader("Upload CSV (Required column: 'smiles')")
        if f: 
            try:
                df_in = pd.read_csv(f)
                df_in.columns = [c.lower().strip() for c in df_in.columns]
                if 'smiles' in df_in.columns:
                    mols = df_in['smiles'].dropna().astype(str).tolist()
                    st.success(f"✅ Successfully loaded {len(mols)} molecules.")
                else:
                    st.error("❌ CSV Error: Missing required column 'smiles'.")
            except Exception as e:
                st.error(f"❌ File Error: {e}")

    if st.button("Start Virtual Screening"):
        if model is None:
            st.error("❌ Action Halted: Model is not loaded.")
        elif not mols:
            st.warning("⚠️ Please provide input data.")
        else:
            # PHASE 1: PREPARATION
            valid_fps = []
            valid_smiles = []
            valid_indices = []
            skipped_count = 0
            
            status_bar = st.progress(0, text="Generating molecular fingerprints...")
            start_time = time.time()
            
            for i, s in enumerate(mols):
                fp = get_fingerprint(s)
                if fp is not None:
                    valid_fps.append(fp)
                    valid_smiles.append(s)
                    valid_indices.append(i)
                else:
                    skipped_count += 1
                
                if i % 10 == 0 or i == len(mols) - 1:
                    status_bar.progress((i + 1) / len(mols), text=f"Processed {i+1}/{len(mols)} molecules")
            
            # PHASE 2: IN
