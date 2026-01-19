import os
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


# =========================
# Page + theme (Gemini-style)
# =========================
st.set_page_config(page_title="NeuroBACE-ML", page_icon="🧠", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

st.sidebar.title("NeuroBACE-ML")
theme_choice = st.sidebar.radio("Appearance Mode", ["Dark", "Light"], horizontal=True)
st.session_state.theme = theme_choice

if st.session_state.theme == "Dark":
    bg, text, card, accent = "#0f172a", "#f8fafc", "#1e293b", "#38bdf8"
    plotly_temp = "plotly_dark"
else:
    bg, text, card, accent = "#ffffff", "#000000", "#f1f5f9", "#2563eb"
    plotly_temp = "plotly_white"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {bg} !important; color: {text} !important; }}
    [data-testid="stSidebar"] {{ background-color: {bg} !important; border-right: 1px solid {accent}33; }}

    h1, h2, h3, h4, label, span, p, [data-testid="stWidgetLabel"] p, .stMarkdown p {{
        color: {text} !important; opacity: 1 !important;
    }}

    [data-testid="stMetric"] {{
        background-color: {card} !important;
        border: 1px solid {accent}44 !important;
        border-radius: 12px;
    }}
    [data-testid="stMetricValue"] div {{ color: {accent} !important; font-weight: bold; }}

    .stButton>button {{
        background: linear-gradient(90deg, #0ea5e9, #2563eb) !important;
        color: white !important; font-weight: bold !important; border-radius: 8px !important;
    }}

    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Files
# =========================
MODEL_JSON = "BACE1_trained_model_optimized.json"
MODEL_PKL = "BACE1_trained_model_optimized.pkl"
FP_BITS = 2048


def here_path(filename: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, filename)


# =========================
# Model loading: JSON Booster first, then PKL fallback
# =========================
@st.cache_resource
def load_model_bundle():
    json_path = here_path(MODEL_JSON)
    pkl_path = here_path(MODEL_PKL)

    if os.path.exists(json_path):
        booster = xgb.Booster()
        booster.load_model(json_path)
        return {"kind": "booster", "model": booster}

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        return {"kind": "pkl", "model": obj}

    return None


def predict_active_prob(model_bundle, X: np.ndarray) -> float:
    """
    - Booster (.json): use Booster.predict(DMatrix) -> probability for binary:logistic
    - PKL (sklearn-style): use predict_proba if available
    """
    kind = model_bundle["kind"]
    model = model_bundle["model"]

    if kind == "booster":
        dmat = xgb.DMatrix(X)
        p = model.predict(dmat)
        return float(p[0])

    # PKL model
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return float(p[0][1])

    # last-resort fallback
    p = model.predict(X)
    return float(p[0])


model_bundle = load_model_bundle()
if model_bundle is None:
    st.error(
        f"Model not found. Upload either {MODEL_JSON} or {MODEL_PKL} to the same folder as app.py in GitHub."
    )
    st.stop()


# =========================
# PubChem: RecordTitle preferred, fallback to IUPACName
# FAST: short timeouts + no retries. Never blocks screening unless user clicks Resolve Names.
# =========================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "NeuroBACE-ML/1.0"})

PUGREST_SMILES_TO_CIDS_JSON = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/cids/JSON"
PUGVIEW_COMPOUND_JSON = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON/?response_type=display"
PUGREST_CID_IUPAC_JSON = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/IUPACName/JSON"
PUGREST_SMILES_TITLE_JSON = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/property/Title/JSON"


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def pubchem_title_from_smiles(smiles: str) -> str:
    try:
        enc = quote(smiles, safe="")
        url = PUGREST_SMILES_TITLE_JSON.format(enc)
        r = SESSION.get(url, timeout=3)
        if r.status_code != 200:
            return "Unknown"
        data = r.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            return "Unknown"
        title = props[0].get("Title")
        return title.strip() if isinstance(title, str) and title.strip() else "Unknown"
    except Exception:
        return "Unknown"


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def pubchem_smiles_to_cid(smiles: str):
    try:
        enc = quote(smiles, safe="")
        url = PUGREST_SMILES_TO_CIDS_JSON.format(enc)
        r = SESSION.get(url, timeout=3)
        if r.status_code != 200:
            return None
        data = r.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        return int(cids[0]) if cids else None
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def pubchem_recordtitle_from_cid(cid: int) -> str:
    try:
        url = PUGVIEW_COMPOUND_JSON.format(cid)
        r = SESSION.get(url, timeout=3)
        if r.status_code != 200:
            return "Unknown"
        data = r.json()
        title = (data.get("Record", {}) or {}).get("RecordTitle")
        if isinstance(title, str) and title.strip():
            return title.strip()
        return "Unknown"
    except Exception:
        return "Unknown"


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def pubchem_iupac_from_cid(cid: int) -> str:
    try:
        url = PUGREST_CID_IUPAC_JSON.format(cid)
        r = SESSION.get(url, timeout=3)
        if r.status_code != 200:
            return "Unknown"
        data = r.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            return "Unknown"
        name = props[0].get("IUPACName")
        return name.strip() if isinstance(name, str) and name.strip() else "Unknown"
    except Exception:
        return "Unknown"


def resolve_name_recordtitle_then_iupac(smiles: str):
    cid = pubchem_smiles_to_cid(smiles)
    if cid is None:
        return "Unknown", None, "None"
    rt = pubchem_recordtitle_from_cid(cid)
    if rt != "Unknown":
        return rt, cid, "RecordTitle"
    iupac = pubchem_iupac_from_cid(cid)
    if iupac != "Unknown":
        return iupac, cid, "IUPACName"
    return "Unknown", cid, "None"


# =========================
# Fingerprinting + prediction
# =========================
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


# =========================
# UI
# =========================
st.title("🧠 NeuroBACE-ML")
st.markdown("##### Advanced platform for BACE1 inhibitor prediction")
st.write("---")

with st.sidebar:
    st.markdown("---")
    threshold = st.slider("Sensitivity Threshold (ACTIVE if ≥ threshold)", 0.0, 1.0, 0.70, 0.01)

    st.subheader("PubChem naming")
    fetch_names = st.checkbox("Enable PubChem naming (optional)", value=False)
    name_method = st.radio(
        "Name method",
        ["Fast Title (1 call)", "RecordTitle then IUPAC (recommended)"],
        index=1,
    )
    resolve_scope = st.radio("Resolve names for", ["Top hits only", "All molecules"], index=0)
    top_n = st.slider("Top N (if Top hits only)", 5, 200, 25, 5)
    st.caption("Tip: Keep naming OFF for fast screening; resolve names after results appear.")

t1, t2, t3 = st.tabs(["🚀 Screening Engine", "📈 Visual Analytics", "🔬 Specifications"])

with t1:
    in_type = st.radio("Input Source", ["Manual Entry", "Batch Upload (CSV)"], horizontal=True)
    mols = []

    if in_type == "Manual Entry":
        raw = st.text_area("SMILES (one per line):", "CC(=O)NC1=CC=C(C=C1)O", height=140)
        mols = [s.strip() for s in raw.split("\n") if s.strip()]
    else:
        f = st.file_uploader("Upload CSV", type=["csv"])
        if f:
            df_in = pd.read_csv(f)
            smi_col = guess_smiles_column(df_in.columns)
            smi_col = st.selectbox("Select SMILES column", options=list(df_in.columns), index=list(df_in.columns).index(smi_col))
            mols = df_in[smi_col].astype(str).tolist()

    c1, c2 = st.columns([1, 1])
    run_btn = c1.button("Start Virtual Screening")
    clear_btn = c2.button("Clear Results")

    if clear_btn:
        st.session_state.pop("results", None)
        st.success("Cleared.")

    # Always render results if present (prevents “nothing happens” feeling)
    if run_btn:
        if not mols:
            st.warning("No SMILES provided.")
        else:
            res = []
            prog = st.progress(0)
            table_ph = st.empty()

            for i, s in enumerate(mols):
                X, err = featurize(s)
                if err:
                    res.append(
                        {
                            "Compound Name": "Unknown",
                            "SMILES": s,
                            "Inhibition Prob": np.nan,
                            "Result": "INVALID",
                            "PubChem CID": None,
                            "Name Source": "None",
                        }
                    )
                else:
                    try:
                        p = predict_active_prob(model_bundle, X)
                        res.append(
                            {
                                "Compound Name": "Unknown",
                                "SMILES": s,
                                "Inhibition Prob": float(round(p, 4)),
                                "Result": "ACTIVE" if p >= threshold else "INACTIVE",
                                "PubChem CID": None,
                                "Name Source": "None",
                        }
                        )
                    except Exception:
                        res.append(
                            {
                                "Compound Name": "Unknown",
                                "SMILES": s,
                                "Inhibition Prob": np.nan,
                                "Result": "ERROR",
                                "PubChem CID": None,
                                "Name Source": "None",
                            }
                        )

                prog.progress((i + 1) / max(1, len(mols)))
                if (i + 1) % 10 == 0 or (i + 1) == len(mols):
                    table_ph.dataframe(pd.DataFrame(res), use_container_width=True)

            df_res = pd.DataFrame(res)
            st.session_state["results"] = df_res

    if "results" in st.session_state:
        df_res = st.session_state["results"].copy()

        valid = df_res[df_res["Result"].isin(["ACTIVE", "INACTIVE"])].dropna(subset=["Inhibition Prob"]).copy()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Molecules", len(df_res))
        m2.metric("Valid predictions", len(valid))
        m3.metric("ACTIVE calls", int((valid["Result"] == "ACTIVE").sum()) if len(valid) else 0)
        m4.metric("Max Probability", f"{valid['Inhibition Prob'].max():.3f}" if len(valid) else "NA")

        st.write("---")
        try:
            st.dataframe(
                df_res.style.background_gradient(subset=["Inhibition Prob"], cmap="RdYlGn"),
                use_container_width=True,
            )
        except Exception:
            st.dataframe(df_res, use_container_width=True)

        st.download_button("Export Results", df_res.to_csv(index=False), "NeuroBACE_Report.csv")

        # Resolve names AFTER predictions (so PubChem never blocks screening)
        if fetch_names:
            if st.button("Resolve Names (PubChem)"):
                with st.spinner("Resolving names from PubChem (short timeouts)…"):
                    df_work = df_res.copy()
                    df_valid = df_work[df_work["Result"].isin(["ACTIVE", "INACTIVE"])].copy()

                    if resolve_scope == "Top hits only":
                        target = df_valid.sort_values("Inhibition Prob", ascending=False).head(top_n)
                        idxs = list(target.index)
                    else:
                        idxs = list(df_valid.index)

                    p2 = st.progress(0)
                    for j, idx in enumerate(idxs):
                        smi = df_work.at[idx, "SMILES"]

                        if name_method.startswith("Fast Title"):
                            nm = pubchem_title_from_smiles(smi)
                            df_work.at[idx, "Compound Name"] = nm
                            df_work.at[idx, "Name Source"] = "Title" if nm != "Unknown" else "None"
                        else:
                            nm, cid, src = resolve_name_recordtitle_then_iupac(smi)
                            df_work.at[idx, "Compound Name"] = nm
                            df_work.at[idx, "PubChem CID"] = cid
                            df_work.at[idx, "Name Source"] = src

                        p2.progress((j + 1) / max(1, len(idxs)))

                    st.session_state["results"] = df_work
                    st.success("Name resolution complete. Scroll up to see updated table.")

with t2:
    if "results" in st.session_state:
        data = st.session_state["results"].copy()
        data_valid = data[data["Result"].isin(["ACTIVE", "INACTIVE"])].dropna(subset=["Inhibition Prob"]).copy()

        st.markdown("### Predictive Probability Distribution")
        if data_valid.empty:
            st.warning("No valid predictions available.")
        else:
            fig_h = px.histogram(
                data_valid,
                x="Inhibition Prob",
                nbins=30,
                template=plotly_temp,
                range_x=[0, 1],
                labels={"Inhibition Prob": "Probability Score"},
            )
            st.plotly_chart(fig_h, use_container_width=True)

            data_sorted = data_valid.sort_values("Inhibition Prob", ascending=True).tail(25)
            fig = px.bar(
                data_sorted,
                y="Compound Name",
                x="Inhibition Prob",
                orientation="h",
                color="Inhibition Prob",
                color_continuous_scale=[[0, "red"], [0.5, "yellow"], [1, "green"]],
                template=plotly_temp,
                labels={"Inhibition Prob": "Probability Score"},
                height=max(400, len(data_sorted) * 30),
            )
            fig.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run screening first to see analytics.")

with t3:
    st.write("### Platform Architecture")
    st.markdown(
        """
- **Model formats supported:** XGBoost native JSON (`Booster.load_model`) or pickled sklearn-style model  
- **Prediction:** Booster uses `predict(DMatrix)` for probabilities in binary logistic setup  
- **Feature Extraction:** 2048-bit Morgan fingerprints (radius = 2)  
- **PubChem naming (optional):**  
  - Fast: SMILES → Title (PUG-REST)  
  - Robust: SMILES → CID (PUG-REST) → RecordTitle (PUG-View) → fallback IUPACName (PUG-REST)  
"""
    )
