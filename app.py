import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NeuroBACE-ML", page_icon="🧠", layout="wide")

# --- HTTP SESSION WITH RETRIES (helps with PubChem throttling/temporary failures) ---
def build_session():
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "NeuroBACE-ML/1.0"})
    return s

SESSION = build_session()

# --- THEME LOGIC ---
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

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("---")
    threshold = st.slider("Probability threshold (ACTIVE if ≥ threshold)", 0.0, 1.0, 0.70, 0.01)
    fetch_names = st.checkbox("Identify compound names from PubChem", value=True)
    show_diag = st.checkbox("Show PubChem diagnostics (CID + source)", value=True)
    if st.button("Clear cached PubChem results"):
        st.cache_data.clear()
        st.success("Cleared PubChem cache. Re-run screening.")

    st.caption("Traffic Light Color Scheme")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    with open("BACE1_trained_model_optimized.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --- PUBCHEM NAME RESOLUTION (RecordTitle preferred; fallback to IUPACName) ---
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)  # cache 7 days
def smiles_to_cid(smiles: str):
    try:
        enc = quote(smiles, safe="")
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{enc}/cids/JSON"
        r = SESSION.get(url, timeout=10)
        if r.status_code != 200:
            return None, f"CID lookup failed (HTTP {r.status_code})"
        data = r.json()
        cid_list = data.get("IdentifierList", {}).get("CID", [])
        if not cid_list:
            return None, "No CID returned"
        return int(cid_list[0]), ""
    except Exception as e:
        return None, f"CID lookup error: {e}"

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def cid_to_recordtitle(cid: int):
    try:
        # PUG-View provides RecordTitle in the JSON under Record.RecordTitle
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/?response_type=display"
        r = SESSION.get(url, timeout=10)
        if r.status_code != 200:
            return None, f"RecordTitle failed (HTTP {r.status_code})"
        data = r.json()
        title = (data.get("Record", {}) or {}).get("RecordTitle")
        if isinstance(title, str) and title.strip():
            return title.strip(), ""
        return None, "RecordTitle missing"
    except Exception as e:
        return None, f"RecordTitle error: {e}"

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def cid_to_iupac(cid: int):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName/JSON"
        r = SESSION.get(url, timeout=10)
        if r.status_code != 200:
            return None, f"IUPAC failed (HTTP {r.status_code})"
        data = r.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            return None, "IUPAC missing"
        name = props[0].get("IUPACName")
        if isinstance(name, str) and name.strip():
            return name.strip(), ""
        return None, "IUPAC missing"
    except Exception as e:
        return None, f"IUPAC error: {e}"

def resolve_name(smiles: str):
    cid, cid_err = smiles_to_cid(smiles)
    if cid is None:
        return "Unknown", None, "None", cid_err

    title, t_err = cid_to_recordtitle(cid)
    if title:
        return title, cid, "RecordTitle", ""

    iupac, i_err = cid_to_iupac(cid)
    if iupac:
        return iupac, cid, "IUPACName", ""

    # If both fail, keep unknown but return diagnostics
    return "Unknown", cid, "None", (t_err or "") + ("; " if t_err and i_err else "") + (i_err or "")

# --- PREDICTION ENGINE ---
def run_prediction(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    try:
        prob = float(model.predict_proba(arr.reshape(1, -1))[0][1])
        return round(prob, 4), ""
    except Exception as e:
        return None, f"Model prediction error: {e}"

# --- MAIN DASHBOARD ---
st.title("NeuroBACE-ML")
st.markdown("##### Advanced platform for BACE1 inhibitor prediction")
st.write("---")

t1, t2, t3 = st.tabs(["Screening Engine", "Visual Analytics", "Specifications"])

with t1:
    in_type = st.radio("Input Source", ["Manual Entry", "Batch Upload (CSV)"], horizontal=True)

    mols = []
    if in_type == "Manual Entry":
        raw = st.text_area("SMILES (one per line):", "CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
        mols = [s.strip() for s in raw.split("\n") if s.strip()]
    else:
        f = st.file_uploader("Upload CSV", type=["csv"])
        if f:
            df_in = pd.read_csv(f)
            cols = list(df_in.columns)
            # Let user choose column (more robust than assuming 'smiles')
            col = st.selectbox("Select the SMILES column", options=cols, index=0)
            mols = df_in[col].astype(str).tolist()

    if st.button("Start Virtual Screening"):
        if not mols:
            st.warning("No SMILES provided.")
        else:
            res = []
            bar = st.progress(0)

            for i, s in enumerate(mols):
                s = str(s).strip()
                prob, perr = run_prediction(s)

                if fetch_names:
                    name, cid, source, nerr = resolve_name(s)
                else:
                    name, cid, source, nerr = "Unknown", None, "Off", ""

                if prob is None:
                    row = {
                        "Compound Name": name,
                        "SMILES": s,
                        "Inhibition Prob": np.nan,
                        "Result": "INVALID" if "Invalid SMILES" in perr else "ERROR",
                        "Error": perr or nerr,
                    }
                else:
                    row = {
                        "Compound Name": name,
                        "SMILES": s,
                        "Inhibition Prob": prob,
                        "Result": "ACTIVE" if prob >= threshold else "INACTIVE",
                        "Error": nerr,
                    }

                if show_diag:
                    row["PubChem CID"] = cid
                    row["Name Source"] = source

                res.append(row)
                bar.progress((i + 1) / len(mols))

            df_res = pd.DataFrame(res)
            st.session_state["results"] = df_res

            valid = df_res[df_res["Result"].isin(["ACTIVE", "INACTIVE"])].copy()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Molecules", len(df_res))
            c2.metric("Valid predictions", len(valid))
            c3.metric("ACTIVE calls", int((valid["Result"] == "ACTIVE").sum()) if len(valid) else 0)
            c4.metric("Max Probability", f"{valid['Inhibition Prob'].max():.2%}" if len(valid) else "NA")

            st.write("---")

            # Color-coded table (traffic light) using pandas style
            if "Inhibition Prob" in df_res.columns:
                st.dataframe(
                    df_res.style.background_gradient(subset=["Inhibition Prob"], cmap="RdYlGn"),
                    use_container_width=True,
                )
            else:
                st.dataframe(df_res, use_container_width=True)

            st.download_button("Export Results", df_res.to_csv(index=False), "NeuroBACE_Report.csv")

with t2:
    if "results" in st.session_state:
        st.markdown("### Predictive Probability Distribution")

        data = st.session_state["results"].copy()
        data_valid = data[data["Result"].isin(["ACTIVE", "INACTIVE"])].dropna(subset=["Inhibition Prob"]).copy()

        if data_valid.empty:
            st.warning("No valid predictions available for analytics.")
        else:
            # Histogram
            fig_h = px.histogram(
                data_valid,
                x="Inhibition Prob",
                nbins=30,
                template=plotly_temp,
                range_x=[0, 1],
                labels={"Inhibition Prob": "Probability Score"},
            )
            st.plotly_chart(fig_h, use_container_width=True)

            # Color-coded horizontal bar (traffic-light scale)
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
                height=max(420, len(data_sorted) * 28),
            )
            fig.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

with t3:
    st.write("### Platform Architecture")
    st.markdown(
        """
- **Model:** XGBoost classifier
- **Feature Extraction:** 2048-bit Morgan fingerprints (radius = 2)
- **Name Resolution (optional):** PubChem PUG-REST (SMILES→CID, IUPACName) and PUG-View (RecordTitle)
"""
    )
