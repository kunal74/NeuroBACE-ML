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
# Page
# -----------------------
st.set_page_config(page_title="NeuroBACE-ML", page_icon="🧠", layout="wide")
st.title("🧠 NeuroBACE-ML")
st.caption("BACE1 inhibition probability prediction for small molecules (SMILES input).")
st.write("---")

# -----------------------
# Constants / files
# -----------------------
MODEL_JSON = "BACE1_trained_model_optimized.json"
MODEL_PKL = "BACE1_trained_model_optimized.pkl"
FP_BITS = 2048

def local_path(filename: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, filename)

# -----------------------
# Model load (JSON Booster first; PKL fallback)
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
    st.error(
        f"Model not found. Upload {MODEL_JSON} (recommended) to the SAME folder as app.py in GitHub."
    )
    st.stop()

def predict_active_prob(model_bundle, X: np.ndarray) -> float:
    """
    Booster (.json): Booster.predict(DMatrix) -> positive-class probability for binary:logistic models.
    PKL (sklearn-like): predict_proba if present.
    """
    kind = model_bundle["kind"]
    model = model_bundle["model"]

    if kind == "booster":
        dmat = xgb.DMatrix(X)
        p = model.predict(dmat)
        return float(p[0])

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return float(p[0][1])

    p = model.predict(X)
    return float(p[0])

# -----------------------
# Featurization (Morgan 2048)
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
    cols = list(columns)
    for c in cols:
        if str(c).strip().lower() in {"smiles", "smile"}:
            return c
    for c in cols:
        if "smiles" in str(c).strip().lower():
            return c
    return cols[0] if cols else None

# -----------------------
# PubChem (fast + robust)
# Uses official PUG-REST + PUG-View endpoints.
# -----------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "NeuroBACE-ML/1.0"})

PUGREST_SMILES_TITLE_JSON = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/property/Title/JSON"
PUGREST_SMILES_TO_CIDS_JSON = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/cids/JSON"
PUGVIEW_COMPOUND_JSON = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON/?response_type=display"
PUGREST_CID_IUPAC_JSON = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/IUPACName/JSON"

def _get_json(url: str, timeout_s: int = 4):
    try:
        r = SESSION.get(url, timeout=timeout_s)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        return r.json(), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def pubchem_title_from_smiles(smiles: str):
    enc = quote(smiles, safe="")
    url = PUGREST_SMILES_TITLE_JSON.format(enc)
    data, err = _get_json(url, timeout_s=4)
    if data is None:
        return None, err
    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return None, "No Title property"
    title = props[0].get("Title")
    if isinstance(title, str) and title.strip():
        return title.strip(), ""
    return None, "Empty Title"

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def pubchem_cid_from_smiles(smiles: str):
    enc = quote(smiles, safe="")
    url = PUGREST_SMILES_TO_CIDS_JSON.format(enc)
    data, err = _get_json(url, timeout_s=4)
    if data is None:
        return None, err
    cids = data.get("IdentifierList", {}).get("CID", [])
    if not cids:
        return None, "No CID returned"
    return int(cids[0]), ""

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def pubchem_recordtitle_from_cid(cid: int):
    url = PUGVIEW_COMPOUND_JSON.format(cid)
    data, err = _get_json(url, timeout_s=4)
    if data is None:
        return None, err
    # PubChem PUG-View example shows Record.RecordTitle. :contentReference[oaicite:3]{index=3}
    title = (data.get("Record", {}) or {}).get("RecordTitle")
    if isinstance(title, str) and title.strip():
        return title.strip(), ""
    return None, "RecordTitle missing"

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def pubchem_iupac_from_cid(cid: int):
    url = PUGREST_CID_IUPAC_JSON.format(cid)
    data, err = _get_json(url, timeout_s=4)
    if data is None:
        return None, err
    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return None, "IUPAC missing"
    name = props[0].get("IUPACName")
    if isinstance(name, str) and name.strip():
        return name.strip(), ""
    return None, "Empty IUPACName"

def resolve_name(smiles: str):
    """
    Prefer Title (fast) -> else CID -> RecordTitle -> else IUPAC.
    Returns: name, cid, source, err
    """
    # 1) Fast Title (Gemini-style)
    title, t_err = pubchem_title_from_smiles(smiles)
    if title:
        return title, None, "Title", ""

    # 2) CID-based
    cid, c_err = pubchem_cid_from_smiles(smiles)
    if cid is None:
        # return the most informative error we have
        err = c_err or t_err or "CID lookup failed"
        return "Unknown", None, "None", err

    rt, rt_err = pubchem_recordtitle_from_cid(cid)
    if rt:
        return rt, cid, "RecordTitle", ""

    iup, i_err = pubchem_iupac_from_cid(cid)
    if iup:
        return iup, cid, "IUPACName", ""

    # Still unknown: provide combined error
    err = "; ".join([x for x in [t_err, rt_err, i_err] if x])
    return "Unknown", cid, "None", err or "No name found"

def pubchem_connectivity_test():
    try:
        r = SESSION.get("https://pubchem.ncbi.nlm.nih.gov/docs/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.subheader("Screening controls")
    threshold = st.slider("ACTIVE if probability ≥", 0.0, 1.0, 0.70, 0.01)

    st.subheader("PubChem naming")
    enable_naming = st.checkbox("Enable PubChem naming (optional)", value=False)
    auto_resolve = st.checkbox("Auto-resolve after screening", value=False)
    resolve_scope = st.radio("Resolve names for", ["Top hits only", "All molecules"], index=0)
    top_n = st.slider("Top N (if Top hits only)", 5, 200, 25, 5)

    if st.button("Test PubChem connectivity"):
        ok = pubchem_connectivity_test()
        if ok:
            st.success("PubChem reachable from this server.")
        else:
            st.error("PubChem NOT reachable from this server.")

    if st.button("Clear PubChem cache"):
        st.cache_data.clear()
        st.success("Cleared PubChem cache.")

# -----------------------
# Tabs
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
        if f is not None:
            df_in = pd.read_csv(f)
            guess = guess_smiles_column(df_in.columns)
            cols = list(df_in.columns)
            idx = cols.index(guess) if guess in cols else 0
            col = st.selectbox("Select SMILES column", options=cols, index=idx)
            smiles_list = df_in[col].astype(str).tolist()

    c1, c2, c3 = st.columns([1, 1, 1])
    run_btn = c1.button("Start Virtual Screening")
    clear_btn = c2.button("Clear Results")
    resolve_btn = c3.button("Resolve Names (PubChem)")

    if clear_btn:
        st.session_state.pop("results", None)
        st.success("Cleared results.")

    # Run screening (FAST: no PubChem calls here)
    if run_btn:
        if not smiles_list:
            st.warning("No SMILES provided.")
        else:
            with st.spinner("Running predictions..."):
                rows = []
                prog = st.progress(0)

                for i, smi in enumerate(smiles_list):
                    smi = str(smi).strip()
                    X, ferr = featurize(smi)

                    if ferr:
                        rows.append({
                            "Compound Name": "Unknown",
                            "SMILES": smi,
                            "Inhibition Prob": np.nan,
                            "Result": "INVALID",
                            "PubChem CID": None,
                            "Name Source": "None",
                            "Name Error": ferr
                        })
                    else:
                        prob = predict_active_prob(model_bundle, X)
                        rows.append({
                            "Compound Name": "Unknown",
                            "SMILES": smi,
                            "Inhibition Prob": float(round(prob, 6)),
                            "Result": "ACTIVE" if prob >= threshold else "INACTIVE",
                            "PubChem CID": None,
                            "Name Source": "None",
                            "Name Error": ""
                        })

                    prog.progress((i + 1) / max(1, len(smiles_list)))

                st.session_state["results"] = pd.DataFrame(rows)

            # Optional auto-resolve (still after results exist)
            if enable_naming and auto_resolve:
                resolve_btn = True

    # Resolve names (only when requested or auto)
    if resolve_btn and enable_naming and "results" in st.session_state:
        df_work = st.session_state["results"].copy()
        df_valid = df_work[df_work["Result"].isin(["ACTIVE", "INACTIVE"])].copy()

        if df_valid.empty:
            st.warning("No valid predictions to resolve.")
        else:
            if resolve_scope == "Top hits only":
                target = df_valid.sort_values("Inhibition Prob", ascending=False).head(top_n)
                idxs = list(target.index)
            else:
                idxs = list(df_valid.index)

            with st.spinner("Resolving names from PubChem (fast Title, then CID/RecordTitle/IUPAC)..."):
                p2 = st.progress(0)
                for j, idx in enumerate(idxs):
                    smi = df_work.at[idx, "SMILES"]
                    name, cid, src, err = resolve_name(smi)
                    df_work.at[idx, "Compound Name"] = name
                    df_work.at[idx, "PubChem CID"] = cid
                    df_work.at[idx, "Name Source"] = src
                    df_work.at[idx, "Name Error"] = err
                    p2.progress((j + 1) / max(1, len(idxs)))
                    time.sleep(0.10)  # small delay to be polite; only affects naming, not screening

            st.session_state["results"] = df_work
            st.success("Name resolution complete. Table updated.")

    # Display results (persisted in session_state)
    if "results" in st.session_state:
        df_out = st.session_state["results"].copy()
        valid = df_out[df_out["Result"].isin(["ACTIVE", "INACTIVE"])].dropna(subset=["Inhibition Prob"]).copy()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Molecules", len(df_out))
        m2.metric("Valid predictions", len(valid))
        m3.metric("ACTIVE calls", int((valid["Result"] == "ACTIVE").sum()) if len(valid) else 0)
        m4.metric("Max Probability", f"{float(valid['Inhibition Prob'].max()):.3f}" if len(valid) else "NA")

        st.write("---")
        try:
            st.dataframe(df_out.style.background_gradient(subset=["Inhibition Prob"], cmap="RdYlGn"),
                         use_container_width=True)
        except Exception:
            st.dataframe(df_out, use_container_width=True)

        st.download_button("Export Results (CSV)", df_out.to_csv(index=False), "NeuroBACE_Report.csv")

    else:
        st.info("Enter SMILES and click Start Virtual Screening.")

with tab2:
    if "results" not in st.session_state:
        st.info("Run screening first to view analytics.")
    else:
        df = st.session_state["results"].copy()
        dfv = df[df["Result"].isin(["ACTIVE", "INACTIVE"])].dropna(subset=["Inhibition Prob"]).copy()

        if dfv.empty:
            st.warning("No valid predictions available.")
        else:
            st.subheader("Predictive Probability Distribution")
            fig_hist = px.histogram(dfv, x="Inhibition Prob", nbins=30, range_x=[0, 1])
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Top predicted molecules (color-coded)")
            top = dfv.sort_values("Inhibition Prob", ascending=False).head(25).copy()
            top = top.sort_values("Inhibition Prob", ascending=True)

            fig_bar = px.bar(
                top,
                y="Compound Name",
                x="Inhibition Prob",
                orientation="h",
                color="Inhibition Prob",
                color_continuous_scale=[[0, "red"], [0.5, "yellow"], [1, "green"]],
                range_x=[0, 1],
                height=max(420, len(top) * 28),
            )
            fig_bar.update_layout(coloraxis_colorbar_title="Probability")
            st.plotly_chart(fig_bar, use_container_width=True)
