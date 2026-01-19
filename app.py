import os
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="NeuroBACE-ML",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroBACE-ML")
st.caption("BACE1 inhibition probability predictor")
st.divider()

# =========================
# CONSTANTS
# =========================
MODEL_FILE = "BACE1_trained_model_optimized.json"
FP_BITS = 2048
THRESHOLD = 0.70

# =========================
# LOAD MODEL (XGBOOST JSON)
# =========================
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), MODEL_FILE)
    booster = xgb.Booster()
    booster.load_model(path)
    return booster

model = load_model()

# =========================
# FEATURE GENERATION
# =========================
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=FP_BITS
    )
    arr = np.zeros((FP_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1)

# =========================
# PREDICTION
# =========================
def predict_probability(X):
    dmat = xgb.DMatrix(X)
    return float(model.predict(dmat)[0])

# =========================
# INPUT
# =========================
smiles_text = st.text_area(
    "Enter SMILES (one per line)",
    "CC(=O)NC1=CC=C(C=C1)O",
    height=150
)

run = st.button("Start Virtual Screening")

# =========================
# SCREENING
# =========================
if run:
    smiles_list = [s.strip() for s in smiles_text.splitlines() if s.strip()]
    results = []

    with st.spinner("Running predictions..."):
        for idx, smi in enumerate(smiles_list, start=1):
            X = featurize(smi)

            if X is None:
                results.append({
                    "Compound ID": f"C-{idx}",
                    "SMILES": smi,
                    "Inhibition Probability": None,
                    "Prediction": "INVALID"
                })
                continue

            prob = predict_probability(X)
            results.append({
                "Compound ID": f"C-{idx}",
                "SMILES": smi,
                "Inhibition Probability": round(prob, 6),
                "Prediction": "ACTIVE" if prob >= THRESHOLD else "INACTIVE"
            })

    df = pd.DataFrame(results)

    # =========================
    # METRICS
    # =========================
    valid = df.dropna(subset=["Inhibition Probability"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Molecules", len(df))
    c2.metric("Valid Predictions", len(valid))
    c3.metric("ACTIVE Calls", int((valid["Prediction"] == "ACTIVE").sum()))
    c4.metric(
        "Max Probability",
        f"{valid['Inhibition Probability'].max():.3f}" if len(valid) else "NA"
    )

    st.divider()

    # =========================
    # RESULTS TABLE
    # =========================
    st.dataframe(
        df.style.background_gradient(
            subset=["Inhibition Probability"],
            cmap="RdYlGn"
        ),
        use_container_width=True
    )

    st.download_button(
        "Download Results (CSV)",
        df.to_csv(index=False),
        "NeuroBACE_BACE1_Screening.csv"
    )
