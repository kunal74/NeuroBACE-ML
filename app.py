import os
import pickle
import time
from typing import Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import xgboost as xgb

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


# =========================
# Configuration
# =========================
MODEL_JSON = "BACE1_trained_model_optimized.json"
MODEL_PKL = "BACE1_trained_model_optimized.pkl"

FP_BITS = 2048
FP_RADIUS = 2

# PubChem endpoints
PUGREST_SMILES_TO_CIDS = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/cids/TXT"
PUGVIEW_COMPOUND_JSON = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON/?response_type=display"
PUGREST_CID_IUPAC_JSON = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/IUPACName/JSON"

# RDKit fingerprint generator
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

# requests session (re-use TCP)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "NeuroBACE-ML/1.0"})


# =========================
# Streamlit page
# =========================
st.set_page_config(page_title="NeuroBACE-ML", layout="wide")

st.title("NeuroBACE-ML: BACE1 Inhibition Probability Predictor")
st.caption("Input SMILES and get an ML-based probability score and hit calling.")

with st.sidebar:
    st.subheader("Controls")
    threshold = st.slider("Probability threshold (ACTIVE if ≥ threshold)", 0.0, 1.0, 0.70, 0.01)
    fetch_names = st.checkbox("Fetch compound names from PubChem", value=True)
    st.caption("If PubChem does not resolve a name, it will remain as Unknown.")


# =========================
# Model loading
# =========================
@st.cache_resource
def load_model_bundle():
    """
    Prefer XGBoost native JSON model for deployment stability.
    Fall back to pickle when JSON is not present.
    """
    if os.path.exists(MODEL_JSON):
        booster = xgb.Booster()
        booster.load_model(MODEL_JSON)
        return {"kind": "booster", "model": booster}

    if os.path.exists(MODEL_PKL):
        with open(MODEL_PKL, "rb") as f:
            obj = pickle.load(f)
        if hasattr(obj, "get_booster"):
            try:
                booster = obj.get_booster()
                return {"kind": "booster", "model": booster}
            except Exception:
                pass
        return {"kind": "pickle", "model": obj}

    return None


def predict_probability(model_bundle, X: np.ndarray) -> float:
    kind = model_bundle["kind"]
    model = model_bundle["model"]

    if kind == "booster":
        dmat = xgb.DMatrix(X)
        p = model.predict(dmat)
        return float(p[0])

    # Fallback for sklearn-like models
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return float(p[0][1])

    if hasattr(model, "predict"):
        p = model.predict(X)
        return float(p[0])

    raise RuntimeError("Loaded model does not support probability prediction.")


model_bundle = load_model_bundle()
if model_bundle is None:
    st.error(
        "Model not found. Upload the model file(s) to the same folder as app.py.\n\n"
        f"Expected: {MODEL_JSON} (preferred) or {MODEL_PKL} (fallback)."
    )
    st.stop()


# =========================
# Chemistry helpers
# =========================
def featurize_smiles(smiles: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"

    try:
        fp = FPGEN.GetFingerprint(mol)
        arr = np.zeros((FP_BITS,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.reshape(1, -1), None
    except Exception as e:
        return None, f"RDKit error: {e}"


def guess_smiles_column(columns) -> Optional[str]:
    # exact common names first
    for c in columns:
        if str(c).strip().lower() in {"smiles", "smile"}:
            return c
    # then partial match
    for c in columns:
        if "smiles" in str(c).strip().lower():
            return c
    return None


# =========================
# PubChem name resolution
# RecordTitle preferred, fallback to IUPACName
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)  # 7 days
def pubchem_smiles_to_cid(smiles: str) -> Optional[int]:
    try:
        enc = quote(smiles, safe="")
        url = PUGREST_SMILES_TO_CIDS.format(enc)
        r = SESSION.get(url, timeout=10)
        if r.status_code != 200:
            return None
        txt = (r.text or "").strip()
        if not txt:
            return None
        # TXT may contain multiple CIDs, take the first
        first = txt.splitlines()[0].strip()
        return int(first)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def pubchem_record_title(cid: int) -> Optional[str]:
    """
    PUG-View compound record typically contains RecordTitle.
    """
    try:
        url = PUGVIEW_COMPOUND_JSON.format(cid)
        r = SESSION.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        title = data.get("RecordTitle")
        if isinstance(title, str) and title.strip():
            return title.strip()
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def pubchem_iupac_name(cid: int) -> Optional[str]:
    try:
        url = PUGREST_CID_IUPAC_JSON.format(cid)
        r = SESSION.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            return None
        name = props[0].get("IUPACName")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return None
    except Exception:
        return None


def resolve_compound_name(smiles: str) -> str:
    """
    Prefer PubChem RecordTitle, fallback to IUPACName.
    If PubChem cannot resolve, return Unknown.
    """
    cid = pubchem_smiles_to_cid(smiles)
    if cid is None:
        return "Unknown"

    title = pubchem_record_title(cid)
    if title:
        return title

    iupac = pubchem_iupac_name(cid)
    if iupac:
        return iupac

    return "Unknown"


# =========================
# UI: Input
# =========================
tab1, tab2, tab3 = st.tabs(["Screening", "Visual analytics", "About"])

with tab1:
    mode = st.radio("Input mode", ["Manual entry", "Batch upload (CSV)"], horizontal=True)

    smiles_list = []
    uploaded_df = None

    if mode == "Manual entry":
        st.write("Enter one SMILES per line.")
        default_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin example
        raw = st.text_area("SMILES", default_smiles, height=140)
        smiles_list = [s.strip() for s in raw.splitlines() if s.strip()]
    else:
        f = st.file_uploader("Upload CSV", type=["csv"])
        if f is not None:
            uploaded_df = pd.read_csv(f)
            guess = guess_smiles_column(uploaded_df.columns)
            cols = list(uploaded_df.columns)
            idx = cols.index(guess) if guess in cols else 0
            col = st.selectbox("Select SMILES column", options=cols, index=idx)
            smiles_list = uploaded_df[col].astype(str).tolist()

    colA, colB = st.columns([1, 1])
    with colA:
        run_btn = st.button("Start screening")
    with colB:
        clear_btn = st.button("Clear results")

    if clear_btn:
        st.session_state.pop("results", None)
        st.success("Cleared.")

    if run_btn:
        if not smiles_list:
            st.warning("No SMILES provided.")
        else:
            rows = []
            prog = st.progress(0)

            # small pause helps avoid PubChem throttling during large batch runs
            # while still feeling responsive.
            for i, s in enumerate(smiles_list):
                s = str(s).strip()
                X, err = featurize_smiles(s)

                if err is not None:
                    name = "Unknown"
                    if fetch_names:
                        name = resolve_compound_name(s)
                    rows.append(
                        {
                            "Compound Name": name,
                            "SMILES": s,
                            "Inhibition Prob": np.nan,
                            "Result": "INVALID",
                            "Error": err,
                        }
                    )
                else:
                    try:
                        prob = predict_probability(model_bundle, X)
                        name = "Unknown"
                        if fetch_names:
                            name = resolve_compound_name(s)

                        rows.append(
                            {
                                "Compound Name": name,
                                "SMILES": s,
                                "Inhibition Prob": float(prob),
                                "Result": "ACTIVE" if prob >= threshold else "INACTIVE",
                                "Error": "",
                            }
                        )
                    except Exception as e:
                        name = "Unknown"
                        if fetch_names:
                            name = resolve_compound_name(s)
                        rows.append(
                            {
                                "Compound Name": name,
                                "SMILES": s,
                                "Inhibition Prob": np.nan,
                                "Result": "ERROR",
                                "Error": str(e),
                            }
                        )

                prog.progress((i + 1) / max(1, len(smiles_list)))

                if fetch_names:
                    time.sleep(0.05)

            df_res = pd.DataFrame(rows)
            st.session_state["results"] = df_res

            valid = df_res[df_res["Result"].isin(["ACTIVE", "INACTIVE"])].copy()
            invalid_or_error = int((df_res["Result"].isin(["INVALID", "ERROR"])).sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total inputs", len(df_res))
            c2.metric("Valid predictions", len(valid))
            c3.metric("ACTIVE calls", int((valid["Result"] == "ACTIVE").sum()) if len(valid) else 0)
            c4.metric("Invalid or errors", invalid_or_error)

            st.subheader("Results")
            st.dataframe(df_res, use_container_width=True)

            st.download_button(
                "Download results CSV",
                data=df_res.to_csv(index=False).encode("utf-8"),
                file_name="NeuroBACE_Report.csv",
                mime="text/csv",
            )


with tab2:
    if "results" not in st.session_state:
        st.info("Run screening first to see analytics.")
    else:
        df = st.session_state["results"].copy()
        valid = df[df["Result"].isin(["ACTIVE", "INACTIVE"])].dropna(subset=["Inhibition Prob"]).copy()

        if valid.empty:
            st.warning("No valid predictions available for analytics.")
        else:
            st.subheader("Predictive Probability Distribution")
            fig_hist = px.histogram(
                valid,
                x="Inhibition Prob",
                nbins=30,
                range_x=[0, 1],
                labels={"Inhibition Prob": "Probability"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Top predicted molecules (color-coded)")
            top_n = min(20, len(valid))
            top = valid.sort_values("Inhibition Prob", ascending=False).head(top_n).copy()

            # bar chart with continuous color scale (red low -> green high)
            top = top.sort_values("Inhibition Prob", ascending=True)
            fig_bar = px.bar(
                top,
                y="Compound Name",
                x="Inhibition Prob",
                orientation="h",
                color="Inhibition Prob",
                range_x=[0, 1],
                color_continuous_scale="RdYlGn",
                labels={"Inhibition Prob": "Probability", "Compound Name": "Compound"},
            )
            fig_bar.update_layout(coloraxis_colorbar_title="Probability")
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("ACTIVE calls only")
            hits = valid[valid["Result"] == "ACTIVE"].sort_values("Inhibition Prob", ascending=False)
            st.dataframe(hits, use_container_width=True)


with tab3:
    st.markdown(
        """
### What the app does
- Converts SMILES into Morgan fingerprints (radius 2, 2048 bits).
- Predicts the probability of BACE1 inhibition using a trained XGBoost model.
- Optionally resolves compound names using PubChem:
  - Prefer **RecordTitle** from PUG-View
  - Fallback to **IUPACName** from PUG-REST

### Notes
- “Probability” is a model score, not experimental potency.
- PubChem naming can return Unknown when no CID is found or when a record has no title.

"""
    )
