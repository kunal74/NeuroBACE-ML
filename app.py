import streamlit as st
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NeuroBACE-ML | BACE1 Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- ADVANCED CUSTOM STYLING (The Beauty Overhaul) ---
st.markdown("""
    <style>
    /* Gradient Header and Background */
    .stApp {
        background: linear-gradient(to bottom, #0f172a, #1e293b);
        color: #f8fafc;
    }
    
    /* Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #334155;
    }
    
    /* Elegant Title and Subheaders */
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    /* Modern Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        color: #ffffff;
    }
    
    /* Remove Metric Boxes Clutter */
    [data-testid="stMetric"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px;
        padding: 20px !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1e293b;
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #38bdf8 !important;
        color: #0f172a !important;
        font-weight: bold;
    }
    
    /* Hide specific Streamlit elements for a custom app feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Fix for broken images */
    img { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: CLEAN CONTROLS ---
st.sidebar.title("NeuroBACE-ML")
st.sidebar.markdown("---")

threshold = st.sidebar.slider(
    "Sensitivity Threshold", 
    0.0, 1.0, 0.70, 0.01,
    help="Adjust probability cut-off for active classification."
)

st.sidebar.markdown("---")
st.sidebar.caption("v1.0.0 | Optimized BACE1 Prediction")

# --- CORE PREDICTION LOGIC ---
@st.cache_resource
def load_optimized_model():
    try:
        with open('BACE1_trained_model_optimized.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_optimized_model()

def predict_compound(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_array = np.array(fp).reshape(1, -1)
        prob = model.predict_proba(fp_array)[0][1]
        return round(prob, 4)
    return None

# --- MAIN DASHBOARD ---
st.title("🧠 NeuroBACE-ML")
st.markdown("#### *High-Fidelity Virtual Screening for Alzheimer's Therapeutic Discovery*")
st.write("---")

tab1, tab2, tab3 = st.tabs(["🚀 Screening Engine", "📈 Results Visualization", "🔬 Platform Data"])

with tab1:
    col_in, col_space = st.columns([2, 1])
    with col_in:
        st.markdown("### Ligand Selection")
        input_type = st.radio("Input Source", ["Manual SMILES Entry", "Batch Dataset Upload (CSV)"], horizontal=True)
        
        compounds = []
        if input_type == "Manual SMILES Entry":
            raw_text = st.text_area("Enter SMILES strings (one per line):", 
                                    "COc1cc2c(cc1OC)C(=O)C(CC2)Cc3ccn(cc3)Cc4ccccc4")
            compounds = [s.strip() for s in raw_text.split('\n') if s.strip()]
        else:
            file = st.file_uploader("Upload CSV (required column: 'smiles')")
            if file:
                df_upload = pd.read_csv(file)
                if 'smiles' in df_upload.columns:
                    compounds = df_upload['smiles'].tolist()
        
        if st.button("Start Virtual Screening"):
            if model and compounds:
                results = []
                for s in compounds:
                    p = predict_compound(s)
                    if p is not None:
                        classification = "ACTIVE" if p >= threshold else "INACTIVE"
                        results.append({"SMILES": s, "Inhibition_Prob": p, "Result": classification})
                
                res_df = pd.DataFrame(results)
                st.session_state['results'] = res_df
                
                st.write("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("Molecules Processed", len(res_df))
                m2.metric("Predicted Actives", len(res_df[res_df['Result'] == "ACTIVE"]))
                m3.metric("Highest Probability", f"{res_df['Inhibition_Prob'].max():.2%}")
                
                st.dataframe(res_df.style.background_gradient(subset=['Inhibition_Prob'], cmap='Blues'), use_container_width=True)
                
                st.download_button("Download Full Report", res_df.to_csv(index=False), "NeuroBACE_Report.csv", "text/csv")
            else:
                st.error("Please provide valid input data.")

with tab2:
    if 'results' in st.session_state:
        st.markdown("### Probability Distribution Analysis")
        data = st.session_state['results']
        fig = px.bar(data, x='SMILES', y='Inhibition_Prob', color='Inhibition_Prob',
                     color_continuous_scale='Blues', template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Execute a screening to view the visual analytics.")

with tab3:
    st.markdown("### Technical Specifications")
    st.write("""
    NeuroBACE-ML is a computational platform built for identifying high-affinity BACE1 inhibitors.
    - **Architecture:** Gradient Boosted Decision Trees (XGBoost).
    - **Optimization:** Bayesian Hyperparameter Tuning.
    - **Precision Score:** 0.8695 | **F1-Score:** 0.8801.
    - **Dataset:** Human BACE1 bioactivity (ChEMBL4822).
    """)
