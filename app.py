import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import base64
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import plotly.express as px
import os
from pathlib import Path
import time

# --- MODEL ARTIFACT (from Colab training) ---
MODEL_FILE = 'BACE1_option1_binary_xgb.json'  # XGBoost Booster JSON
DEFAULT_THRESHOLD = 0.70                     # finalized operating threshold
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / MODEL_FILE

# --- APPLICABILITY DOMAIN (AD) SETTINGS ---
TRAIN_REF_FILE = 'training_reference_smiles.csv'  # placed next to app.py
AD_DEFAULT_SIM_CUT = 0.40
AD_DEFAULT_MARGIN  = 0.20
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
    threshold = st.slider("Operating Threshold (finalized default)", 0.0, 1.0, float(DEFAULT_THRESHOLD), 0.01)
    st.caption("v1.1 | High-Performance Batching")

    st.markdown("### Reliability controls")
    enable_ad = st.toggle("Enable Applicability Domain + Abstain", value=True)
    sim_cut = st.slider("AD similarity cutoff (Max Tanimoto)", 0.0, 1.0, float(AD_DEFAULT_SIM_CUT), 0.01, disabled=not enable_ad)
    margin = st.slider("Abstain margin around threshold", 0.0, 0.5, float(AD_DEFAULT_MARGIN), 0.01, disabled=not enable_ad)


# --- PREDICTION ENGINE ---
@st.cache_resource
def load_model(model_path: str, mtime: float):
    """Load and cache the trained XGBoost Booster.
    mtime invalidates the cache when the model file is updated.
    """
    if not os.path.exists(model_path):
        st.error(f"🚨 Critical Error: Model file is missing. Expected at: {model_path}")
        return None
    try:
        booster = xgb.Booster()
        booster.load_model(model_path)
        return booster
    except xgb.core.XGBoostError as e:
        st.error(f"🚨 Model Error: {e}")
        return None
    except Exception as e:
        st.error(f"🚨 Unexpected Error: {e}")
        return None

model = load_model(str(MODEL_PATH), os.path.getmtime(str(MODEL_PATH)) if os.path.exists(str(MODEL_PATH)) else 0.0)

# Load AD reference fingerprints (if file exists)


mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
# --- APPLICABILITY DOMAIN (AD) HELPERS ---
@st.cache_resource
def load_reference_fps(ref_csv_path: str):
    """Load reference Morgan fingerprints for AD similarity (cached)."""
    import os
    if not os.path.exists(ref_csv_path):
        return []
    df_ref = pd.read_csv(ref_csv_path)
    df_ref.columns = [c.lower().strip() for c in df_ref.columns]
    col = "smiles" if "smiles" in df_ref.columns else ("smiles_std" if "smiles_std" in df_ref.columns else None)
    if col is None:
        return []
    fps = []
    for s in df_ref[col].dropna().astype(str).tolist():
        mol = Chem.MolFromSmiles(s)
        if mol:
            fps.append(mfpgen.GetFingerprint(mol))
    return fps

def fp_bitvect_to_numpy(fp_bv, n_bits: int = 2048):
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp_bv, arr)
    return arr

def max_tanimoto(query_fp_bv, ref_fps):
    if not ref_fps:
        return None
    sims = DataStructs.BulkTanimotoSimilarity(query_fp_bv, ref_fps)
    return float(max(sims)) if sims else None


# Load AD reference fingerprints (if file exists)
ref_fps = load_reference_fps(str(BASE_DIR / TRAIN_REF_FILE))  # [] if missing
# --- SCIENTIFIC HELPER FUNCTIONS ---
def get_confidence_level(prob: float, thr: float) -> str:
    """Confidence based on distance from the operating threshold (thr)."""
    dist = abs(float(prob) - float(thr))
    if dist < 0.05:
        return "LOW"
    elif dist < 0.15:
        return "MEDIUM"
    else:
        return "HIGH"

# REFACTORED: Single molecule processor (Generate Fingerprint Only)
def get_fingerprint(smiles):
    try:
        if not smiles or not isinstance(smiles, str):
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Explicit Bit-Vector Generation (Scientific Correctness)
            fp_bitvect = mfpgen.GetFingerprint(mol)
            
            # Create a fixed-size numpy array for the bit vector
            fp = np.zeros((2048,), dtype=np.int8) 
            DataStructs.ConvertToNumpyArray(fp_bitvect, fp)
            return fp
    except:
        return None
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

# --- TAB 1: SCREENING ENGINE (VECTORIZED) ---
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
            # PHASE 1: PREPARATION (Vectorized Fingerprint Generation)
            valid_fps = []
            valid_smiles = []
            valid_indices = []
            valid_maxsim = []
            skipped_count = 0
            
            status_bar = st.progress(0, text="Generating molecular fingerprints...")
            
            start_time = time.time()
            
            for i, s in enumerate(mols):
                try:
                    mol = Chem.MolFromSmiles(s)
                    if not mol:
                        skipped_count += 1
                        continue

                    fp_bv = mfpgen.GetFingerprint(mol)
                    fp_np = fp_bitvect_to_numpy(fp_bv, n_bits=2048)

                    # AD similarity (optional)
                    ms = None
                    if 'enable_ad' in locals() and enable_ad:
                        ms = max_tanimoto(fp_bv, ref_fps)

                    valid_fps.append(fp_np)
                    valid_smiles.append(s)
                    valid_indices.append(i)
                    valid_maxsim.append(ms)
                except Exception:
                    skipped_count += 1
                
                # Update bar occasionally to avoid UI lag
                if i % 10 == 0 or i == len(mols) - 1:
                    status_bar.progress((i + 1) / len(mols), text=f"Processed {i+1}/{len(mols)} molecules")
            
            # PHASE 2: INFERENCE (Single Batch Call)
            if valid_fps:
                status_bar.progress(0.9, text="Running XGBoost inference engine...")
                
                # Stack all fingerprints into a single 2D Numpy Matrix (N x 2048)
                X_batch = np.vstack(valid_fps)
                
                # Create ONE DMatrix for the whole batch
                dmatrix_batch = xgb.DMatrix(X_batch)
                
                # Predict ONCE
                probs_batch = model.predict(dmatrix_batch)
                
                # Map results back
                res = []
                for idx, (smile, p) in enumerate(zip(valid_smiles, probs_batch)):
                    prob_val = float(p)
                    ms = valid_maxsim[idx] if idx < len(valid_maxsim) else None

                    ad_status = "AD_DISABLED"
                    decision = "ACCEPT"
                    if 'enable_ad' in locals() and enable_ad and ms is not None:
                        if ms < sim_cut:
                            ad_status = "OUTSIDE_AD"
                            decision = "ABSTAIN"
                        elif abs(prob_val - threshold) < margin:
                            ad_status = "BORDERLINE"
                            decision = "ABSTAIN"
                        else:
                            ad_status = "IN_AD"
                            decision = "ACCEPT"

                    if decision == "ABSTAIN":
                        final_call = "ABSTAIN"
                    else:
                        final_call = "ACTIVE" if prob_val >= threshold else "INACTIVE"

                    res.append({
                        "Compounds": f"C-{valid_indices[idx]+1}",
                        "Inhibition Prob": round(prob_val, 4),
                        "MaxSim": None if ms is None else round(ms, 3),
                        "AD_Status": ad_status,
                        "Decision": decision,
                        "Model Confidence": get_confidence_level(prob_val, threshold),
                        "Result": final_call,
                        "SMILES": smile
                    })
                
                status_bar.empty() # Remove progress bar on completion
                
                # Calculate processing time
                total_time = time.time() - start_time
                st.toast(f"Screening complete in {total_time:.2f} seconds!", icon="🚀")

                if skipped_count > 0:
                    st.warning(f"⚠️ Note: {skipped_count} molecule(s) skipped due to invalid structure.")

                df_res = pd.DataFrame(res)
                st.session_state['results'] = df_res
                
                c1_m, c2_m, c3_m = st.columns(3)
                c1_m.metric("Molecules Processed", len(df_res))
                c2_m.metric("Potent Hits", len(df_res[df_res['Result'] == "ACTIVE"]))
                c3_m.metric("Max Probability", f"{df_res['Inhibition Prob'].max():.2%}")
                
                st.write("---")
                
                def highlight_low_conf(row):
                    if "LOW" in row['Model Confidence']:
                        return ['background-color: #3f3f3f; color: #ffa500'] * len(row)
                    return [''] * len(row)

                # --- Traffic-light cell coloring (no matplotlib required) ---
                prob_col = "Inhibition Prob" if "Inhibition Prob" in df_res.columns else ("P(Active)" if "P(Active)" in df_res.columns else None)

                def traffic_light(v):
                    try:
                        v = float(v)
                    except Exception:
                        return ""
                    if v >= 0.90:
                        return "background-color: #16a34a; color: white;"   # green
                    elif v >= 0.70:
                        return "background-color: #65a30d; color: white;"   # light green
                    elif v >= 0.50:
                        return "background-color: #facc15; color: black;"   # yellow
                    elif v >= 0.30:
                        return "background-color: #fb923c; color: black;"   # orange
                    else:
                        return "background-color: #dc2626; color: white;"   # red

                if prob_col:
                    styled = (
                        df_res.style
                        .applymap(traffic_light, subset=[prob_col])
                        .format({prob_col: "{:.4f}"})
                    )
                    st.dataframe(styled, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(df_res, use_container_width=True, hide_index=True)

                st.download_button("Export Results", df_res.to_csv(index=False), "NeuroBACE_Report.csv")
            else:
                st.error("❌ No valid fingerprints could be generated.")

# --- VISUAL ANALYTICS ---
with t2:
    if 'results' in st.session_state:
        st.markdown("### Predictive Probability Distribution")
        data = st.session_state['results'].sort_values('Inhibition Prob', ascending=True)
        
        # Keep plot colors consistent with the table "traffic light" rules by fixing the color range to [0, 1]
        # and using the same breakpoints: <0.30 red, 0.30–0.50 orange, 0.50–0.70 yellow, 0.70–0.90 light green, ≥0.90 green.
        traffic_scale = [
            [0.00, "#dc2626"], [0.2999, "#dc2626"],  # red
            [0.30, "#fb923c"], [0.4999, "#fb923c"],  # orange
            [0.50, "#facc15"], [0.6999, "#facc15"],  # yellow
            [0.70, "#65a30d"], [0.8999, "#65a30d"],  # light green
            [0.90, "#16a34a"], [1.00, "#16a34a"],    # green
        ]

        fig = px.bar(
            data,
            y="Compounds",
            x="Inhibition Prob",
            orientation="h",
            color="Inhibition Prob",
            range_color=[0, 1],
            color_continuous_scale=traffic_scale,
            template=plotly_temp,
            hover_data=["Model Confidence", "SMILES"],
            labels={"Inhibition Prob": "Probability Score"},
            height=max(400, len(data) * 30)
        )

        fig.update_layout(xaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

# --- SPECIFICATIONS ---
with t3:
    st.write("### Platform Architecture")
    st.markdown("""
- **Inference Engine:** XGBoost binary classifier (native JSON serialization)
- **Processing:** Vectorized batch inference for high-throughput screening
- **Molecular Encoding:** RDKit Morgan fingerprints (radius 2, 2048-bit explicit bit vector)
- **Optimization:** Bayesian hyperparameter tuning with Optuna (offline training phase)
- **Scientific Validation:** Scaffold-split evaluation (multi-seed) and Y-randomization sanity check

**Reliability Controls**
- **Confidence Estimation:** Probability-margin heuristic based on \|P(active) − threshold\|
- **Applicability Domain (AD):** Max Tanimoto similarity to the training reference set is reported (MaxSim)
- **Abstain Policy (optional):** Abstain when (i) MaxSim < 0.40 or (ii) \|P(active) − threshold\| < 0.20

**Labeling and Decision**
- **Labeling (strict):** Active if pIC50 ≥ 7; Inactive if pIC50 ≤ 6; gray zone excluded (6 < pIC50 < 7)
- **Operating Threshold:** Default 0.70 (user adjustable)

**Identification**
- **Local Serial Nomenclature:** C-n
""")
