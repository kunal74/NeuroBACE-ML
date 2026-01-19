import os
import time
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st
import xgboost as xgb
import plotly.express as px

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


# =========================
# Config
# =========================
MODEL_JSON = "BACE1_trained_model_optimized.json"
FP_BITS = 2048
FP_RADIUS = 2

# PubChem endpoints (official)
PUGREST_SMILES_TO_CIDS_JSON = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/cids/JSON"
PUGVIEW_COMPOUND_JSON = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON/?response_type=display"
PUGREST_CID_IUPAC_JSON = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/IUPACName/JSON"


# =========================
# Streamlit setup
# =========================
st.set_page_config(page_title="NeuroBACE-ML", layout="wide")

st.title("NeuroBACE-ML: BACE1 Inhibition Probability Predictor")
st.caption("Input SMILES and receive a probability score from the trained model. Optional PubChem naming is supported.")


# =========================
# Networking: requests session with retries
# =========================
def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "NeuroBACE-ML/1.0"})
    return s


SESSION = _build_session()


# =========================
# Model loading (JSON Booster)
# =========================
@st.cache_resource
def load_booster() -> xgb.Booster:
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(here, MODEL_JSON)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_JSON}. Upload it to the same folder as app.py in GitHub."
        )

    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def predict_active_prob(booster: xgb.Booster, X: np.ndarray) -> float:
    """
    For a binary logistic XGBoost Booster, booster.predict returns the positive-class probability.
    Uses DMatrix for compatibility.
    """
    dmat = xgb.DMatrix(X)
    p = booster.predict(dmat)
    return float(p[0])


# =========================
# RDKit fingerprinting
# =========================
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def featurize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"

    try:
        fp = FPGEN.GetFingerprint(mol)
        arr = np.zeros((FP_BITS,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.reshape(1, -1), ""
    except Exception as e:
        return None, f"RDKit error: {e}"


def guess_smiles_column(columns):
    # Exact common names
    for c in columns:
        if str(c).strip().lower() in {"smiles", "smile"}:
            return c
    # Partial match
    for c in columns:
        if "smiles" in str(c).strip().lower():
            return c
    return None


# =========================
# PubChem naming: RecordTitle preferred, fallback IUPACName
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def pubchem_smiles_to_cid(smiles: str):
    try:
        enc = quote(smiles, safe="")
        url = PUGREST_SMILES_TO_CIDS_JSON.format(enc)
        r = SESSION.get(url, timeout=12)
        if r.status_code != 200:
            return None, f"CID lookup failed (HTTP {r.status_code})"
        data = r.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            return None, "No CID returned"
        return int(cids[0]), ""
    except Exception as e:
        return None, f"CID lookup error: {e}"


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def pubchem_record_title(cid: int):
    """
    PUG-View returns JSON where RecordTitle is typically under Record.RecordTitle.
    """
    try:
        url = PUGVIEW_COMPOUND_JSON.format(cid)
        r = SESSION.get(url, timeout=12)
        if r.status_code != 200:
            return None, f"RecordTitle failed (HTTP {r.status_code})"
        data = r.json()

        # Most common structure
        title = (data.get("Record", {}) or {}).get("RecordTitle")

        # Fallback (some representations)
        if not title:
            title = data.get("RecordTitle")

        if isinstance(title, str) and title.strip():
            return title.strip(), ""
        return None, "RecordTitle missing"
    except Exception as e:
        return None, f"RecordTitle error: {e}"


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def pubchem_iupac_name(cid: int):
    try:
        url = PUGREST_CID_IUPAC_JSON.format(cid)
        r = SESSION.get(url, timeout=12)
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


def resolve_compound_name(smiles: str):
    """
    Returns:
      name, cid, source, error
    """
    cid, cid_err = pubchem_smiles_to_cid(smiles)
    if cid is None:
        return "Unknown", None, "None", cid_err

    title, t_err = pubchem_record_title(cid)
    if title:
        return title, cid, "RecordTitle", ""

    iupac, i_err = pubchem_iupac_name(cid)
    if iupac:
        return iupac, cid, "IUPACName", ""

    err = "; ".join([x for x in [t_err, i_err] if x])
    return "Unknown", cid, "None", err


# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.subheader("Screening Controls")
    threshold = st.slider("Probability threshold (ACTIVE if ≥ threshold)", 0.0, 1.0, 0.70, 0.01)

    st.subheader("Compound Naming (PubChem)")
    fetch_names = st.checkbox("Identify compounds from SMILES (PubChem)", value=True)

    name_mode = st.radio(
        "Resolve names for",
        ["Top hits only (recommended)", "All molecules", "Off"],
        index=0,
    )
    top_n = st.slider("Top N for name resolution", 5, 100, 25, 5)

    st.caption("Tip: If screening feels slow, set naming to Top hits only or Off.")

    if st.button("Clear PubChem cache"):
        st.cache_data.clear()
        st.success("Cleared cached PubChem results. Re-run screening.")


# =========================
# Main tabs
# =========================
tab1, tab2, tab3 = st.tabs(["Screening", "Visual analytics", "Specifications"])

booster = None
try:
    booster = load_booster()
except Exception as e:
    st.error(str(e))
    st.stop()


# =========================
# Screening tab
# =========================
with tab1:
    input_mode = st.radio("Input source", ["Manual entry", "Batch upload (CSV)"], horizontal=True)

    smiles_list = []
    if input_mode == "Manual entry":
        st.write("Enter one SMILES per line.")
        default_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin example
        raw = st.text_area("SMILES (one per line)", default_smiles, height=140)
        smiles_list = [s.strip() for s in raw.splitlines() if s.strip()]
    else:
        f = st.file_uploader("Upload CSV file", type=["csv"])
        if f is not None:
            df_in = pd.read_csv(f)
            guess = guess_smiles_column(df_in.columns)
            cols = list(df_in.columns)
            idx = cols.index(guess) if guess in cols else 0
            col = st.selectbox("Select the SMILES column", options=cols, index=idx)
            smiles_list = df_in[col].astype(str).tolist()

    cA, cB = st.columns([1, 1])
    run = cA.button("Start Virtual Screening")
    clear = cB.button("Clear Results")

    if clear:
        st.session_state.pop("results", None)
        st.success("Results cleared.")

    # Always display results if present (prevents “nothing happens” after reruns)
    results_placeholder = st.empty()

    if run:
        if not smiles_list:
            st.warning("No SMILES provided.")
        else:
            with st.spinner("Running predictions..."):
                rows = []
                prog = st.progress(0)
                table_ph = st.empty()

                for i, smi in enumerate(smiles_list):
                    smi = str(smi).strip()
                    X, ferr = featurize_smiles(smi)

                    if ferr:
                        rows.append(
                            {
                                "Compound Name": "Unknown",
                                "SMILES": smi,
                                "Inhibition Prob": np.nan,
                                "Result": "INVALID",
                                "PubChem CID": None,
                                "Name Source": "None",
                                "Error": ferr,
                            }
                        )
                    else:
                        try:
                            prob = predict_active_prob(booster, X)
                            rows.append(
                                {
                                    "Compound Name": "Unknown",
                                    "SMILES": smi,
                                    "Inhibition Prob": float(prob),
                                    "Result": "ACTIVE" if prob >= threshold else "INACTIVE",
                                    "PubChem CID": None,
                                    "Name Source": "None",
                                    "Error": "",
                                }
                            )
                        except Exception as e:
                            rows.append(
                                {
                                    "Compound Name": "Unknown",
                                    "SMILES": smi,
                                    "Inhibition Prob": np.nan,
                                    "Result": "ERROR",
                                    "PubChem CID": None,
                                    "Name Source": "None",
                                    "Error": f"Model prediction error: {e}",
                                }
                            )

                    # Update progress + occasional live table
                    prog.progress((i + 1) / max(1, len(smiles_list)))
                    if (i + 1) % 10 == 0 or (i + 1) == len(smiles_list):
                        table_ph.dataframe(pd.DataFrame(rows), use_container_width=True)

                df_res = pd.DataFrame(rows)

            # Optional: PubChem naming AFTER predictions (prevents long “no output” time)
            if fetch_names and name_mode != "Off":
                with st.spinner("Resolving compound names via PubChem..."):
                    df_valid = df_res[df_res["Result"].isin(["ACTIVE", "INACTIVE"])].copy()

                    if name_mode.startswith("Top"):
                        df_target = df_valid.sort_values("Inhibition Prob", ascending=False).head(top_n)
                        target_idx = set(df_target.index.tolist())
                    else:
                        target_idx = set(df_valid.index.tolist())

                    p2 = st.progress(0)
                    done = 0
                    total = len(target_idx)

                    # PubChem rate: keep it polite (avoid throttling)
                    # PubChem guidance commonly recommends low request rates for large runs.
                    for idx in target_idx:
                        smi = df_res.at[idx, "SMILES"]
                        name, cid, src, nerr = resolve_compound_name(smi)
                        df_res.at[idx, "Compound Name"] = name
                        df_res.at[idx, "PubChem CID"] = cid
                        df_res.at[idx, "Name Source"] = src
                        # keep name-resolution errors visible (but do not overwrite existing model errors)
                        if nerr and not df_res.at[idx, "Error"]:
                            df_res.at[idx, "Error"] = nerr

                        done += 1
                        p2.progress(done / max(1, total))
                        time.sleep(0.25)  # ~4 requests/sec to reduce throttling

            st.session_state["results"] = df_res

    # Render results if available
    if "results" in st.session_state:
        df_out = st.session_state["results"].copy()
        valid = df_out[df_out["Result"].isin(["ACTIVE", "INACTIVE"])].dropna(subset=["Inhibition Prob"]).copy()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Molecules", len(df_out))
        m2.metric("Valid predictions", len(valid))
        m3.metric("ACTIVE calls", int((valid["Result"] == "ACTIVE").sum()) if len(valid) else 0)
        m4.metric("Max Probability", f"{float(valid['Inhibition Prob'].max()):.3f}" if len(valid) else "NA")

        st.write("---")

        # Color-coded table (traffic light style)
        try:
            styled = df_out.style.format({"Inhibition Prob": "{:.4f}"}).background_gradient(
                subset=["Inhibition Prob"], cmap="RdYlGn"
            )
            results_placeholder.dataframe(styled, use_container_width=True)
        except Exception:
            results_placeholder.dataframe(df_out, use_container_width=True)

        st.download_button(
            "Export Results (CSV)",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="NeuroBACE_Report.csv",
            mime="text/csv",
        )


# =========================
# Visual analytics tab
# =========================
with tab2:
    if "results" not in st.session_state:
        st.info("Run screening first to see analytics.")
    else:
        df = st.session_state["results"].copy()
        dfv = df[df["Result"].isin(["ACTIVE", "INACTIVE"])].dropna(subset=["Inhibition Prob"]).copy()

        if dfv.empty:
            st.warning("No valid predictions available for analytics.")
        else:
            st.subheader("Predictive Probability Distribution")
            fig_hist = px.histogram(
                dfv,
                x="Inhibition Prob",
                nbins=30,
                range_x=[0, 1],
                labels={"Inhibition Prob": "Probability Score"},
            )
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
                labels={"Inhibition Prob": "Probability Score", "Compound Name": "Compound"},
                height=max(420, len(top) * 28),
            )
            fig_bar.update_layout(coloraxis_colorbar_title="Probability")
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("ACTIVE calls only")
            hits = dfv[dfv["Result"] == "ACTIVE"].sort_values("Inhibition Prob", ascending=False)
            st.dataframe(hits, use_container_width=True)


# =========================
# Specifications tab
# =========================
with tab3:
    st.markdown(
        """
### Platform summary
- **Model format:** XGBoost native JSON (`Booster.load_model`)  
- **Prediction:** `Booster.predict(DMatrix)` returns the positive-class score for binary logistic models.  
- **Featurization:** Morgan fingerprints (radius 2, 2048 bits)  
- **Compound naming:** PubChem
  - SMILES → CID using **PUG-REST**
  - CID → RecordTitle using **PUG-View**
  - Fallback CID → IUPACName using **PUG-REST**

### Notes
- The output is an ML probability score, not experimental potency.
- PubChem naming may return Unknown for compounds without resolvable CIDs or during throttling.
"""
    )
