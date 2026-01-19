import os
import time
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import xgboost as xgb
from urllib.parse import quote
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# -----------------------
# 1. SETUP & CORE UTILITIES
# -----------------------
st.set_page_config(page_title="NeuroBACE-ML", page_icon="🧠", layout="wide")

# This function was missing in your original code, causing a crash
def pubchem_connectivity_test():
    try:
        response = requests.get("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/C/description/JSON", timeout=2)
        return response.status_code == 200
    except:
        return False

# This function was named differently in your code and calling a missing variable
@st.cache_data(show_spinner=False)
def resolve_name_and_cid(smiles):
    try:
        escaped_smiles = quote(smiles)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{escaped_smiles}/property/Title,CID/JSON"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            props = data['PropertyTable']['Properties'][0]
            return props.get('Title', "Unknown"), props.get('CID', "N/A")
        return "Not Found", "N/A"
    except:
        return "Connection Error", "N/A"

# -----------------------
# 2. MODEL & FEATURIZATION
# -----------------------
MODEL_JSON = "BACE1_trained_model_optimized.json"
FP_BITS = 2048

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_JSON):
        model = xgb.Booster()
        model.load_model(MODEL_JSON)
        return model
    return None

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
    arr = np.zeros((1,), dtype=np.float32)
    arr = np.zeros((FP_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1)

# -----------------------
# 3. USER INTERFACE
# -----------------------
st.title("🧠 NeuroBACE-ML")
st.write("---")

with st.sidebar:
    st.subheader("Settings")
    threshold = st.slider("Activity Threshold", 0.0, 1.0, 0.70)
    enable_naming = st.checkbox("Fetch PubChem Names", value=True)
    
    if st.button("Check PubChem Status"):
        if pubchem_connectivity_test(): st.success("PubChem Online")
        else: st.error("PubChem Offline/Blocked")

# Input Section
raw_input = st.text_area("Enter SMILES (one per line)", "CC(=O)NC1=CC=C(C=C1)O")
smiles_list = [s.strip() for s in raw_input.splitlines() if s.strip()]

if st.button("🚀 Start Virtual Screening"):
    model = load_model()
    if model is None:
        st.error("Model file not found. Please upload BACE1_trained_model_optimized.json")
    else:
        results = []
        progress_bar = st.progress(0)
        
        for i, smi in enumerate(smiles_list):
            X = featurize(smi)
            if X is not None:
                # Prediction
                dmat = xgb.DMatrix(X)
                prob = float(model.predict(dmat)[0])
                
                # Naming Logic (The Fix)
                name, cid = "Unknown", "N/A"
                if enable_naming:
                    name, cid = resolve_name_and_cid(smi)
                
                results.append({
                    "Compound Name": name,
                    "SMILES": smi,
                    "Inhibition Prob": round(prob, 4),
                    "Result": "ACTIVE" if prob >= threshold else "INACTIVE",
                    "PubChem CID": cid
                })
            progress_bar.progress((i + 1) / len(smiles_list))
        
        # Display Results
        df = pd.DataFrame(results)
        st.session_state["my_results"] = df
        
        # Formatting the table for clarity
        st.subheader("Screening Results")
        st.dataframe(
            df.style.background_gradient(subset=["Inhibition Prob"], cmap="RdYlGn"),
            use_container_width=True
        )
