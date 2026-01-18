import streamlit as st
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import plotly.express as px
import requests
from streamlit_lottie import st_lottie

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NeuroBACE-ML", page_icon="🧠", layout="wide")

# --- ADAPTIVE THEME LOGIC ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'

st.sidebar.title("NeuroBACE-ML")
theme_choice = st.sidebar.radio("Appearance Mode", ["Dark", "Light"], horizontal=True)
st.session_state.theme = theme_choice

# Define Theme Variables for Perfect Visibility
if st.session_state.theme == 'Dark':
    bg, text, card, accent = "#0f172a", "#f8fafc", "#1e293b", "#38bdf8"
    plotly_temp = "plotly_dark"
else:
    # High-contrast light theme: Black text on white background
    bg, text, card, accent = "#ffffff", "#000000", "#f1f5f9", "#2563eb"
    plotly_temp = "plotly_white"

# --- UNIVERSAL VISIBILITY CSS ---
st.markdown(f"""
    <style>
    /* Global Background and Text */
    .stApp {{ background-color: {bg} !important; color: {text} !important; }}
    
    /* Force Sidebar Visibility and Text Color */
    [data-testid="stSidebar"] {{ background-color: {bg} !important; border-right: 1px solid {accent}33; }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{ color: {text} !important; font-weight: 500; }}
    
    /* Header & Widget Label Visibility */
    h1, h2, h3, h4, label, .stMarkdown p, [data-testid="stWidgetLabel"] p {{ 
        color: {text} !important; 
        opacity: 1 !important; 
    }}
    
    /* Remove Metric Blocks and Apply Themed Cards */
    [data-testid="stMetric"] {{ 
        background-color: {card}
