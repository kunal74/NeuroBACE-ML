import os
import time
import pickle
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import xgboost as xgb

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(page_title="NeuroBACE-ML", page_icon="🧠", layout="wide")
st.title("🧠 NeuroBACE-ML")
st.caption("BACE1 inhibition probability prediction for small molecules (SMILES input).")
st.write("---")

# -----------------------
# Constants / Files
# -----------------------
MODEL_JSON = "BACE1_trained_model_optimized.json"
MODEL_PKL = "BACE1_trained_model_optimized.pkl"
FP_BITS = 2048

def local_path(filename: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, filename)

# -----------------------
# Utility: PubChem & Naming (FIXED)
# -----------------------
@st.cache_data(show_spinner=False)
def resolve_name(smiles):
    """Fetches Compound Name and CID from PubChem REST API."""
    try:
        escaped_smiles = quote(smiles)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{escaped_smiles}/property/Title,CID/JSON"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            props = response.json()['PropertyTable']['Properties'][0]
            cid = props.get('CID', None)
            name = props.get('Title', "Unknown Ligand")
            return name, cid, "PubChem", ""
        return "Unknown", None, "None", "Molecule not found in PubChem"
    except Exception as e:
        return "Novel Molecule", None, "None", str(e)

def pubchem_connectivity_test():
    """Checks if the server can reach PubChem."""
    try:
        response = requests.get("https://pubchem.ncbi.nlm.nih.gov", timeout=3)
        return response.status_code == 200
    except:
        return False

# -----------------------
# Model Loading
# -----------------------
@st.cache_resource
def load_model_bundle():
    json_path = local_path(MODEL_JSON)
    pkl_path = local_path(MODEL_PKL)

    if os.path.exists(json_path):
        booster = xgb.Booster()
        booster.load_model(json_path)
        return {"kind": "booster", "model": booster, "path": json_path}

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        return {"kind": "pkl", "model": obj, "path": pkl_path}
    return None

model_bundle = load_model_bundle()
if model_bundle is None:
    st.error(f"Model not found. Ensure {MODEL_JSON} is in the same directory.")
    st.stop()

def predict_active_prob(model_bundle, X: np.ndarray) -> float:
    kind = model_bundle["kind"]
    model = model_bundle["model"]
    if kind == "booster":
        dmat = xgb.DMatrix(X)
        p = model.predict(dmat)
        return float(p[0])
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return float(p[0][1])
    return float(model.predict(X)[0])

# -----------------------
# Featurization
# -----------------------
def featurize(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
    arr = np.zeros((FP_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1), ""

def guess_smiles_column(columns):
    for c in columns:
        if str(c).strip().lower() in {"smiles", "smile"}: return c
    return columns[0] if len(columns) > 0 else None

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.subheader("Screening controls")
    threshold = st.slider("ACTIVE if probability ≥", 0.0, 1.0, 0.70, 0.01)

    st.subheader("PubChem naming")
    enable_naming = st.checkbox("Enable PubChem naming", value=True)
    auto_resolve = st.checkbox("Auto-resolve after screening", value=True)
    resolve_scope = st.radio("Resolve names for", ["Top hits only", "All molecules"], index=0)
    top_n = st.slider("Top N hits", 5, 200, 25, 5)

    if st.button("Test PubChem connectivity"):
        if pubchem_connectivity_test():
            st.success("PubChem reachable.")
        else:
            st.error("PubChem NOT reachable.")

# -----------------------
# Main Logic
# -----------------------
tab1, tab2 = st.tabs(["🚀 Screening", "📈 Visual analytics"])

with tab1:
    input_mode = st.radio("Input source", ["Manual entry", "Batch upload (CSV)"], horizontal=True)
    smiles_list = []

    if input_mode == "Manual entry":
        raw = st.text_area("SMILES (one per line)", "CC(=O)NC1=CC=C(C=C1)O", height=140)
        smiles_list = [s.strip() for s in raw.splitlines() if s.strip()]
    else:
        f = st.file_uploader("Upload CSV", type=["csv"])
        if f:
            df_in = pd.read_csv(f)
            col = st.selectbox("Select SMILES column", options=df_in.columns, index=0)
            smiles_list = df_in[col].astype(str).tolist()

    run_btn = st.button("Start Virtual Screening")
    
    if run_btn and smiles_list:
        rows = []
        prog = st.progress(0)
        for i, smi in enumerate(smiles_list):
            X, ferr = featurize(smi)
            if ferr:
                rows.append({"Compound Name": "Error", "SMILES": smi, "Inhibition Prob": np.nan, "Result": "INVALID", "Name Error": ferr})
            else:
                prob = predict_active_prob(model_bundle, X)
                # Initial name is Unknown; will be updated if auto_resolve is on
                rows.append({
                    "Compound Name": "Unknown", 
                    "SMILES": smi, 
                    "Inhibition Prob": round(prob, 6), 
                    "Result": "ACTIVE" if prob >= threshold else "INACTIVE",
                    "PubChem CID": None, "Name Source": "None", "Name Error": ""
                })
            prog.progress((i + 1) / len(smiles_list))
        
        st.session_state["results"] = pd.DataFrame(rows)

        # TRIGGER AUTO-RESOLVE
        if enable_naming and auto_resolve:
            df_work = st.session_state["results"]
            valid_indices = df_work.index[df_work["Result"] != "INVALID"].tolist()
            
            if resolve_scope == "Top hits only":
                valid_indices = df_work.loc[valid_indices].sort_values("Inhibition Prob", ascending=False).head(top_n).index.tolist()
            
            with st.spinner("Fetching names..."):
                for idx in valid_indices:
                    name, cid, src, err = resolve_name(df_work.at[idx, "SMILES"])
                    df_work.at[idx, "Compound Name"] = name
                    df_work.at[idx, "PubChem CID"] = cid
                    df_work.at[idx, "Name Source"] = src
                    df_work.at[idx, "Name Error"] = err
            st.session_state["results"] = df_work

    if "results" in st.session_state:
        st.dataframe(st.session_state["results"].style.background_gradient(subset=["Inhibition Prob"], cmap="RdYlGn"), use_container_width=True)
