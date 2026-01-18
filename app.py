import streamlit as st
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NeuroBACE-ML | BACE1 Predictive Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM THEMING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6; }
    .stButton>button { width: 100%; border-radius: 4px; background-color: #003366; color: white; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://img.icons8.com/external-flatart-icons-flat-flatarticons/64/external-brain-biotechnology-flatart-icons-flat-flatarticons.png", width=80)
st.sidebar.title("NeuroBACE-ML")
st.sidebar.divider()

# Probability Threshold (Reference: AlzyFinder confidence > 0.70)
threshold = st.sidebar.slider(
    "Activity Probability Threshold", 
    min_value=0.01, max_value=1.0, value=0.70, step=0.01,
    help="Compounds with a probability score above this value are classified as 'Potent Inhibitors'."
)

st.sidebar.markdown("---")
st.sidebar.info("Lead Scientist: Dr. Kunal Bhattacharya\nSpecialization: CADD & Neuropharmacology")

# --- MODEL LOADER ---
@st.cache_resource
def load_neurobace_model():
    try:
        # Loading the optimized XGBoost brain
        with open('BACE1_trained_model_optimized.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Ensure 'BACE1_trained_model_optimized.pkl' is in the repository.")
        return None

model = load_neurobace_model()

# --- PREDICTION ENGINE ---
def run_prediction(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Generating 2048-bit Morgan Fingerprints (ECFP4 equivalent)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_array = np.array(fp).reshape(1, -1)
        prob = model.predict_proba(fp_array)[0][1]
        return round(prob, 4)
    return None

# --- MAIN INTERFACE ---
st.title("🧠 NeuroBACE-ML")
st.subheader("Advanced Machine Learning Platform for BACE1 Inhibitor Prediction")
st.write("Targeting Beta-secretase 1 (BACE1) for novel Alzheimer's Disease therapeutic discovery.")

tab1, tab2, tab3 = st.tabs(["🔍 Screening Engine", "📊 Visualization", "📄 Documentation"])

with tab1:
    st.write("### Ligand Input")
    input_mode = st.radio("Select Input Type:", ["Single/Multiple SMILES", "Batch CSV Upload"])
    
    compounds = []
    
    if input_mode == "Single/Multiple SMILES":
        user_text = st.text_area("Enter SMILES (one per line):", 
                                "COc1cc2c(cc1OC)C(=O)C(CC2)Cc3ccn(cc3)Cc4ccccc4")
        compounds = [s.strip() for s in user_text.split('\n') if s.strip()]
    else:
        uploaded_csv = st.file_uploader("Upload CSV (must contain a 'smiles' column):")
        if uploaded_csv:
            df_in = pd.read_csv(uploaded_csv)
            if 'smiles' in df_in.columns:
                compounds = df_in['smiles'].tolist()
            else:
                st.error("Column 'smiles' not found in the uploaded file.")

    if st.button("🚀 Execute NeuroBACE-ML Screening"):
        if model and compounds:
            results = []
            for sm in compounds:
                p = run_prediction(sm)
                if p is not None:
                    status = "POTENT" if p >= threshold else "WEAK/INACTIVE"
                    results.append({"SMILES": sm, "Probability": p, "Classification": status})
            
            res_df = pd.DataFrame(results)
            st.session_state['res_df'] = res_df
            
            # Overview Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Molecules Screened", len(res_df))
            m2.metric("Potent Hits (≥0.70)", len(res_df[res_df['Probability'] >= 0.70]))
            m3.metric("Highest Hit Probability", f"{res_df['Probability'].max():.2%}")
            
            st.divider()
            st.write("### Screening Results Table")
            st.dataframe(res_df, use_container_width=True)
            
            # Export Result (Standard for Journals)
            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Results (CSV)", csv_data, "NeuroBACE_ML_Results.csv", "text/csv")
        else:
            st.warning("Please provide SMILES and ensure the model file is uploaded.")

with tab2:
    if 'res_df' in st.session_state:
        st.write("### Predictive Heatmap")
        plot_df = st.session_state['res_df']
        fig = px.bar(plot_df, x='SMILES', y='Probability', color='Probability',
                     color_continuous_scale='Viridis', title="Probability Distribution of Screened Ligands")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Visual distribution of inhibition probabilities. Higher values indicate stronger BACE1 interaction potential.")
    else:
        st.info("Please run a screening in the first tab to generate visualizations.")

with tab3:
    st.write("### Platform Specifications")
    st.markdown("""
    **NeuroBACE-ML** is a specialized virtual screening tool built upon the **AlzyFinder** methodological framework[cite: 528, 563]:
    
    * **Core Algorithm:** eXtreme Gradient Boosting (XGBoost)[cite: 531].
    * **Optimization:** Automated Bayesian hyperparameter tuning via Optuna[cite: 535].
    * **Molecular Features:** 2048-bit Morgan Fingerprints (Radius = 2)[cite: 645].
    * **Dataset:** Curated from ChEMBL v32, specifically targeting Human BACE1 (CHEMBL4822)[cite: 579, 602].
    * **Validation:** * **Balanced Accuracy:** 0.86
        * **F1-Score:** 0.88
        * **Precision:** 0.87
    """)
    st.divider()
    st.markdown("Developed for the research community as an open-access resource for Alzheimer's drug discovery.")