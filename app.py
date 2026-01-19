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
from rdkit.Chem import inchi

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================
# Page / theme
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
    h1,h2,h3,h4,label,span,p,[data-testid="stWidgetLabel"] p,.stMarkdown p {{ color: {text} !important; opacity: 1 !important; }}
    [data-testid="stMetric"] {{ background-color: {card} !important; border: 1px solid {accent}44 !important; border-radius: 12px; }}
    [data-testid="stMetricValue"] div {{ color: {accent} !important; font-weight: bold; }}
    .stButton>button {{ background: linear-gradient(90deg, #0ea5e9, #2563eb) !important; color: white !important; font-weight: bold !important; border-radius: 8px !important; }}
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 NeuroBACE-ML")
st.markdown("##### Advanced platform for BACE1 inhibitor prediction")
st.write("---")


# =========================
# Paths / constants
# =========================
MODEL_JSON = "BACE1_trained_model_optimized.json"
MODEL_PKL = "BACE1_trained_model_optimized.pkl"
FP_BITS = 2048

def here_path(filename: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, filename)


# =========================
# Model loading (JSON Booster preferred, PKL fallback)
# XGBoost Booster uses predict(DMatrix), not predict_proba. :contentReference[oaicite:4]{index=4}
# =========================
@st.cache_resource
def load_model_bundle():
    json_path = here_path(MODEL_JSON)
    pkl_path = here_path(MODEL_PKL)

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
    st.error(f"Model not found. Put {MODEL_JSON} (or {MODEL_PKL}) in the same folder as app.py.")
    st.stop()

def predict_active_prob(bundle, X: np.ndarray) -> float:
    kind = bundle["kind"]
    model = bundle["model"]

    if kind == "booster":
        dmat = xgb.DMatrix(X)
        p = model.predict(dmat)
        return float(p[0])

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return float(p[0][1])

    p = model.predict(X)
    return float(p[0])


# =========================
# RDKit featurization
# =========================
def featurize(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, "Invalid SMILES"

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
    arr = np.zeros((FP_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    # InChIKey is used as a PubChem fallback lookup
    try:
        ik = inchi.MolToInchiKey(mol)
    except Exception:
        ik = None

    return arr.reshape(1, -1), ik, ""


def guess_smiles_column(cols):
    cols = list(cols)
    for c in cols:
        if str(c).strip().lower() in {"smiles", "smile"}:
            return c
    for c in cols:
        if "smiles" in str(c).strip().lower():
            return c
    return cols[0] if cols else None


# =========================
# PubChem utilities (robust)
# Uses official PUG-REST endpoints. :contentReference[oaicite:5]{index=5}
# Retries configured via urllib3 Retry. :contentReference[oaicite:6]{index=6}
# =========================
@st.cache_resource
def pubchem_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "NeuroBACE-ML/1.0"})
    retry = Retry(
        total=4,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

PUBCHEM = pubchem_session()

PUG_TITLE_BY_SMILES = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/property/Title/JSON"
PUG_TITLE_BY_INCHIKEY = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{}/property/Title/JSON"

def pubchem_title_from_smiles(smiles: str, timeout_s: int = 6):
    enc = quote(smiles, safe="")
    url = PUG_TITLE_BY_SMILES.format(enc)
    try:
        r = PUBCHEM.get(url, timeout=timeout_s)
        if r.status_code == 200:
            data = r.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            title = props[0].get("Title") if props else None
            if isinstance(title, str) and title.strip():
                return title.strip(), "", "Title"
            return None, "No Title returned (200 OK)", "None"
        return None, f"HTTP {r.status_code}", "None"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", "None"

def pubchem_title_from_inchikey(inchikey: str, timeout_s: int = 6):
    if not inchikey:
        return None, "Missing InChIKey", "None"
    enc = quote(inchikey, safe="")
    url = PUG_TITLE_BY_INCHIKEY.format(enc)
    try:
        r = PUBCHEM.get(url, timeout=timeout_s)
        if r.status_code == 200:
            data = r.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            title = props[0].get("Title") if props else None
            if isinstance(title, str) and title.strip():
                return title.strip(), "", "Title(InChIKey)"
            return None, "No Title returned (200 OK)", "None"
        return None, f"HTTP {r.status_code}", "None"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", "None"

def resolve_pubchem_name(smiles: str, inchikey: str):
    # 1) Try SMILES -> Title
    name, err, src = pubchem_title_from_smiles(smiles)
    if name:
        return name, src, ""

    # 2) Fallback: InChIKey -> Title
    name2, err2, src2 = pubchem_title_from_inchikey(inchikey)
    if name2:
        return name2, src2, ""

    # Return best error message
    msg = err or err2 or "Name lookup failed"
    return "Unknown", "None", msg


# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.markdown("---")
    threshold = st.slider("Sensitivity Threshold (ACTIVE if ≥)", 0.0, 1.0, 0.70, 0.01)

    st.subheader("PubChem naming (optional)")
    enable_naming = st.checkbox("Enable PubChem naming", value=False)
    resolve_scope = st.radio("Resolve names for", ["Top hits only", "All molecules"], index=0)
    top_n = st.slider("Top N (if Top hits only)", 5, 200, 25, 5)
    throttle = st.slider("Throttle between PubChem calls (sec)", 0.0, 1.0, 0.25, 0.05)

    if st.button("Clear PubChem cache"):
        st.cache_data.clear()
        st.success("Cleared PubChem cache.")


# =========================
# Tabs
# =========================
t1, t2 = st.tabs(["🚀 Screening Engine", "📈 Visual Analytics"])

with t1:
    in_type = st.radio("Input Source", ["Manual Entry", "Batch Upload (CSV)"], horizontal=True)
    mols = []

    if in_type == "Manual Entry":
        raw = st.text_area("SMILES (one per line):", "CC(=O)NC1=CC=C(C=C1)O", height=140)
        mols = [s.strip() for s in raw.splitlines() if s.strip()]
    else:
        f = st.file_uploader("Upload CSV", type=["csv"])
        if f is not None:
            df_in = pd.read_csv(f)
            guess = guess_smiles_column(df_in.columns)
            cols = list(df_in.columns)
            idx = cols.index(guess) if guess in cols else 0
            col = st.selectbox("Select the SMILES column", options=cols, index=idx)
            mols = df_in[col].astype(str).tolist()

    c1, c2, c3 = st.columns([1, 1, 1])
    run_btn = c1.button("Start Virtual Screening")
    clear_btn = c2.button("Clear Results")
    resolve_btn = c3.button("Resolve Names (PubChem)")

    if clear_btn:
        st.session_state.pop("results", None)
        st.success("Results cleared.")

    # Screening (FAST: no PubChem calls)
    if run_btn:
        if not mols:
            st.warning("No SMILES provided.")
        else:
            with st.spinner("Running predictions..."):
                res = []
                prog = st.progress(0)

                for i, s in enumerate(mols):
                    X, ik, ferr = featurize(s)

                    if ferr:
                        res.append({
                            "Compound Name": "Unknown",
                            "SMILES": s,
                            "Inhibition Prob": np.nan,
                            "Result": "INVALID",
                            "Name Source": "None",
                            "Name Error": ferr,
                            "InChIKey": ik or ""
                        })
                    else:
                        p = predict_active_prob(model_bundle, X)
                        res.append({
                            "Compound Name": "Unknown",
                            "SMILES": s,
                            "Inhibition Prob": float(round(p, 6)),
                            "Result": "ACTIVE" if p >= threshold else "INACTIVE",
                            "Name Source": "None",
                            "Name Error": "Not resolved",
                            "InChIKey": ik or ""
                        })

                    prog.progress((i + 1) / max(1, len(mols)))

                st.session_state["results"] = pd.DataFrame(res)

    # Name resolution (only on click)
    if resolve_btn:
        if not enable_naming:
            st.warning("Enable PubChem naming in the sidebar first.")
        elif "results" not in st.session_state:
            st.warning("Run screening first.")
        else:
            df_work = st.session_state["results"].copy()
            df_valid = df_work[df_work["Result"].isin(["ACTIVE", "INACTIVE"])].copy()

            if df_valid.empty:
                st.warning("No valid predictions to resolve.")
            else:
                if resolve_scope == "Top hits only":
                    idxs = list(df_valid.sort_values("Inhibition Prob", ascending=False).head(top_n).index)
                else:
                    idxs = list(df_valid.index)

                with st.spinner("Resolving names from PubChem (with retry/backoff)..."):
                    p2 = st.progress(0)
                    for j, idx in enumerate(idxs):
                        smi = df_work.at[idx, "SMILES"]
                        ik = df_work.at[idx, "InChIKey"]
                        name, src, err = resolve_pubchem_name(smi, ik)

                        df_work.at[idx, "Compound Name"] = name
                        df_work.at[idx, "Name Source"] = src
                        df_work.at[idx, "Name Error"] = err if err else ""

                        p2.progress((j + 1) / max(1, len(idxs)))
                        if throttle > 0:
                            time.sleep(throttle)

                st.session_state["results"] = df_work
                st.success("Name resolution finished (see Name Error for any PubChem issues).")

    # Display results
    if "results" in st.session_state:
        df_res = st.session_state["results"].copy()
        valid = df_res[df_res["Result"].isin(["ACTIVE", "INACTIVE"])].dropna(subset=["Inhibition Prob"]).copy()

       m1, m2, m3, m4 = st.columns(4)
       m1.metric("Molecules", len(df_res))
       m2.metric("Valid predictions", len(valid))
       m3.metric("ACTIVE calls", int((valid["Result"] == "ACTIVE").sum()) if len(valid) else 0)

       max_prob = float(valid["Inhibition Prob"].max()) if len(valid) else None
       m4.metric("Max Probability", f"{max_prob:.3f}" if max_prob is not None else "NA")

        st.write("---")
        try:
            st.dataframe(df_res.style.background_gradient(subset=["Inhibition Prob"], cmap="RdYlGn"),
                         use_container_width=True)
        except Exception:
            st.dataframe(df_res, use_container_width=True)

        st.download_button("Export Results", df_res.to_csv(index=False), "NeuroBACE_Report.csv")
    else:
        st.info("Enter SMILES and click Start Virtual Screening.")

with t2:
    if "results" not in st.session_state:
        st.info("Run screening first to see analytics.")
    else:
        df = st.session_state["results"].copy()
        dfv = df[df["Result"].isin(["ACTIVE", "INACTIVE"])].dropna(subset=["Inhibition Prob"]).copy()

        if dfv.empty:
            st.warning("No valid predictions available.")
        else:
            st.subheader("Predictive Probability Distribution")
            fig_hist = px.histogram(dfv, x="Inhibition Prob", nbins=30, range_x=[0, 1], template=plotly_temp)
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
                template=plotly_temp,
            )
            fig_bar.update_layout(coloraxis_colorbar_title="Probability")
            st.plotly_chart(fig_bar, use_container_width=True)
