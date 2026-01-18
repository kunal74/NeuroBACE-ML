import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NeuroBACE-ML", page_icon="🧠", layout="wide")

# --- CUSTOM COLORMAP DEFINITION ---
# Creates a smooth transition: Orange (Lowest) -> Yellow -> Green (Highest)
custom_oyg_cmap = LinearSegmentedColormap.from_list("oyg", ["#ff9900", "#ffff00", "#00cc00"])
plotly_oyg = ["#ff9900", "#ffff00", "#00cc00"]

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

st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg} !important; color: {text} !important; }}
    [data-testid="stSidebar"] {{ background-color: {bg} !important; border-right: 1px solid {accent}33; }}
    h1, h2, h3, h4, label, span, p, [data-testid="stWidgetLabel"] p, .stMarkdown p {{ 
        color: {text} !important; opacity: 1 !important; 
    }}
    [data-testid="stMetric"] {{ background-color: {card} !important; border-radius: 12px; border: 1px solid {accent}44; }}
    [data-testid="stMetricValue"] div {{ color: {accent} !important; font-weight: bold; }}
    .stButton>button {{ background: linear-gradient(90deg, #0ea5e9, #2563eb) !important; color: white !important; font-weight: bold !important; border-radius: 8px !important; }}
    img {{ display: none !important; }}
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

# --- UTILITY: PUBCHEM NAME RECOGNITION ---
def get_compound_name(smiles):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/Title/JSON"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()['PropertyTable']['Properties'][0].get('Title', "Unknown")
        return "Novel/Unknown"
    except:
        return "Fetch Error"

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("---")
    threshold = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.70, 0.01)
    st.caption("v1.2.0 | Aesthetic Color Update")

# --- PREDICTION ENGINE ---
@st.cache_resource
def load_model():
    try:
        with open('BACE1_trained_model_optimized.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

model = load_model()

def run_prediction(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return round(model.predict_proba(np.array(fp).reshape(1, -1))[0][1], 4)
    return None

# --- MAIN DASHBOARD ---
st.title("🧠 NeuroBACE-ML")
st.markdown("##### *Advanced Machine Learning Platform for BACE1 Inhibitor Prediction*")
st.write("---")

t1, t2, t3 = st.tabs(["🚀 Screening Engine", "📈 Visual Analytics", "🔬 Specifications"])

with t1:
    in_type = st.radio("Input Source", ["Manual Entry", "Batch Upload (CSV)"], horizontal=True)
    mols = []
    if in_type == "Manual Entry":
        raw = st.text_area("SMILES (one per line):", "COc1cc2c(cc1OC)C(=O)C(CC2)Cc3ccn(cc3)Cc4ccccc4")
        mols = [s.strip() for s in raw.split('\n') if s.strip()]
    else:
        f = st.file_uploader("Upload Dataset")
        if f: 
            df_in = pd.read_csv(f)
            mols = df_in['smiles'].tolist() if 'smiles' in df_in.columns else []

    if st.button("Start Virtual Screening"):
        if model and mols:
            res = []
            bar = st.progress(0)
            for i, s in enumerate(mols):
                p = run_prediction(s)
                if p is not None:
                    res.append({
                        "Compound Name": get_compound_name(s),
                        "SMILES": s, 
                        "Prob": p, 
                        "Result": "ACTIVE" if p >= threshold else "INACTIVE"
                    })
                bar.progress((i + 1) / len(mols))
            
            df_res = pd.DataFrame(res)
            st.session_state['results'] = df_res
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Processed", len(df_res))
            c2.metric("Potent Hits", len(df_res[df_res['Result'] == "ACTIVE"]))
            c3.metric("Max Probability", f"{df_res['Prob'].max():.2%}")
            
            st.write("---")
            # Apply Orange-Yellow-Green Gradient to Table
            st.dataframe(df_res.style.background_gradient(subset=['Prob'], cmap=custom_oyg_cmap), use_container_width=True)
            st.download_button("📥 Export Results (CSV)", df_res.to_csv(index=False), "NeuroBACE_OYG_Report.csv")
        else:
            st.error("Please provide valid data.")

with t2:
    if 'results' in st.session_state:
        # Apply Orange-Yellow-Green Gradient to Bar Chart
        fig = px.bar(st.session_state['results'], x='Compound Name', y='Prob', color='Prob', 
                     color_continuous_scale=plotly_oyg, template=plotly_temp)
        st.plotly_chart(fig, use_container_width=True)

with t3:
    st.write("### Platform Specifications")
    st.markdown(f"""
    - **Architecture:** Optimized XGBoost Model
    - **Precision:** 0.8695 | **F1 Score:** 0.8801
    - **Balanced Accuracy:** 0.8619
    - **Dataset:** 8,750 Curated BACE1 Bioactivity Records
    """)
