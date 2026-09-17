"""Streamlit web interface for Bird Eco DIP - Styled Edition."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from config import OUTPUTS_DIR, IMAGES_DIR, MODELS_DIR, DATA_DIR
from src.preprocessing import preprocess
from src.segmentation import segment
from src.features import extract_all_features
from src.classifier import predict_species

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Bird Eco DIP",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS ====================
st.markdown("""
<style>
    /* Sidebar dark background */
    [data-testid="stSidebar"] {
        background-color: #1a2f4a !important;
    }
    
    /* ALL text in sidebar white */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Radio buttons specifically */
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
        font-size: 15px !important;
    }
    
    /* Radio button circles */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        color: white !important;
    }
    
    /* Main background */
    .main .block-container {
        background-color: #f5f7fa;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 4px solid #2ecc71;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #1a2f4a;
    }
    .metric-label {
        font-size: 13px;
        color: #666;
        margin-top: 5px;
    }
    
    /* Feature badges */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
    }
    .feature-badge {
        background: white;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        text-align: center;
    }
    .feature-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    .feature-name {
        font-size: 12px;
        color: #888;
    }
    .feature-value {
        font-size: 18px;
        font-weight: bold;
        color: #1a2f4a;
    }
    
    /* Eco cards */
    .eco-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
    }
    .eco-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .eco-icon {
        font-size: 24px;
        margin-bottom: 5px;
    }
    .eco-label {
        font-size: 11px;
        color: #888;
    }
    .eco-value {
        font-size: 16px;
        font-weight: bold;
        color: #1a2f4a;
    }
    
    /* Section cards */
    .section-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 16px;
    }
    .section-title {
        font-size: 16px;
        font-weight: bold;
        color: #1a2f4a;
        margin-bottom: 12px;
    }
    
    /* Prediction card */
    .pred-card {
        background: linear-gradient(135deg, #1a2f4a 0%, #2c3e50 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .pred-name {
        font-size: 20px;
        font-weight: bold;
    }
    .pred-confidence {
        font-size: 14px;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
@st.cache_data
def load_features():
    return pd.read_csv(os.path.join(OUTPUTS_DIR, "features.csv"))

@st.cache_data
def load_eco_lookup():
    return pd.read_csv(os.path.join(OUTPUTS_DIR, "eco_lookup.csv"))

@st.cache_data
def load_pca():
    return pd.read_csv(os.path.join(OUTPUTS_DIR, "pca_results.csv"))

features_df = load_features()
eco_df = load_eco_lookup()
pca_df = load_pca()

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()
# Initialize session state for recent uploads
if "recent_images" not in st.session_state:
    st.session_state.recent_images = []

# ==================== HELPERS ====================
def get_image_path(filename):
    """Find image path in dataset folders."""
    matches = list(Path(IMAGES_DIR).rglob(filename))
    if matches:
        return str(matches[0])
    return None

def find_nearest_birds(target_pc1, target_pc2, n=10):
    """Find n nearest birds in PCA space."""
    dists = np.sqrt((pca_df["PC1"] - target_pc1)**2 + (pca_df["PC2"] - target_pc2)**2)
    idx = dists.nsmallest(n).index
    return pca_df.iloc[idx].copy()

def metric_card(value, label, icon=""):
    html = f"""
    <div class="metric-card">
        <div style="font-size:28px;margin-bottom:5px;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def feature_badge(name, value, color):
    html = f"""
    <div class="feature-badge">
        <span class="feature-dot" style="background:{color};"></span>
        <span class="feature-name">{name}</span>
        <div class="feature-value">{value}</div>
    </div>
    """
    return html

def eco_card(icon, label, value, unit):
    html = f"""
    <div class="eco-card">
        <div class="eco-icon">{icon}</div>
        <div class="eco-label">{label}</div>
        <div class="eco-value">{value} <span style="font-size:11px;color:#888;">{unit}</span></div>
    </div>
    """
    return html

# ==================== SIDEBAR ====================
with st.sidebar:
    # Logo - solo el logo, sin texto adicional porque ya incluye el nombre
    logo_path = os.path.join("assets", "logo1.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown("<h1 style='color:white;text-align:center;'>🐦</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:white;text-align:center;'>Bird Eco DIP</h3>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🖼️ Process Image", "📈 PCA Analysis"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<p style='color:#888;font-size:11px;text-align:center;'>© 2026 Bird Eco DIP</p>", unsafe_allow_html=True)

# ==================== PAGE 1: DASHBOARD ====================
if page == "📊 Dashboard":
    st.markdown("<h1 style='color:#1a2f4a;'>Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666;'>Overview of bird image analysis</p>", unsafe_allow_html=True)
    
    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("11,788", "Images Processed", "🖼️")
    with c2:
        metric_card("200", "Species", "🐦")
    with c3:
        metric_card("7", "Features Extracted", "📊")
    with c4:
        metric_card("10,538", "AVONET Species", "🌿")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts and recent images
    left, right = st.columns([3, 2])
    
    with left:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📊 Circularity Distribution</div>
        </div>
        """, unsafe_allow_html=True)
        st.bar_chart(features_df["circularity"], use_container_width=True)
    
    with right:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🕐 Recent Uploads</div>
        </div>
        """, unsafe_allow_html=True)
        
        if len(st.session_state.recent_images) == 0:
            st.info("No images uploaded yet. Go to **Process Image** to analyze birds.")
        else:
            for item in st.session_state.recent_images[:5]:
                cols = st.columns([1, 3])
                with cols[0]:
                    st.image(item["image"], width=50)
                with cols[1]:
                    st.markdown(f"<small><b>{item['filename'][:30]}</b></small>", unsafe_allow_html=True)
                    st.markdown(f"<small style='color:#888;'>Area: {item['area']:,.0f} px | Circ: {item['circularity']:.3f}</small>", unsafe_allow_html=True)
                st.markdown("<hr style='margin:5px 0;opacity:0.3;'>", unsafe_allow_html=True)
# ==================== PAGE 2: PROCESS IMAGE ====================
elif page == "🖼️ Process Image":
    st.markdown("<h1 style='color:#1a2f4a;'>Process Image</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666;'>Upload a bird image to analyze</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        left_col, right_col = st.columns([2, 3])
        
        with left_col:
            st.markdown("""
            <div class="section-card">
                <div class="section-title">🖼️ Original</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with right_col:
            # Process
            gray, enhanced = preprocess(img)
            mask, bbox, contour = segment(enhanced)
            
            if mask is not None:
                # Segmentation card
                st.markdown("""
                <div class="section-card">
                    <div class="section-title">✂️ Segmentation</div>
                </div>
                """, unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
                with c2:
                    st.image(mask, caption="Mask", use_container_width=True)
                
                # Features card
                feats = extract_all_features(img, gray, mask, contour)
                                # Save to recent history
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.session_state.recent_images.insert(0, {
                    "image": img_rgb,
                    "filename": uploaded_file.name,
                    "area": feats["area"],
                    "circularity": feats["circularity"],
                    "symmetry": feats["symmetry"],
                })
                # Keep only last 10
                st.session_state.recent_images = st.session_state.recent_images[:10]
                
                st.markdown("""
                <div class="section-card">
                    <div class="section-title">📊 Visual Features</div>
                </div>
                """, unsafe_allow_html=True)
                
                f1, f2, f3 = st.columns(3)
                with f1:
                    st.markdown(feature_badge("Area", f"{feats['area']:,.0f} px", "#3498db"), unsafe_allow_html=True)
                with f2:
                    st.markdown(feature_badge("Circularity", f"{feats['circularity']:.3f}", "#2ecc71"), unsafe_allow_html=True)
                with f3:
                    st.markdown(feature_badge("Symmetry", f"{feats['symmetry']:.3f}", "#f39c12"), unsafe_allow_html=True)
                
                f4, f5, f6 = st.columns(3)
                with f4:
                    st.markdown(feature_badge("Fractal Dim", f"{feats['fractal_dim']:.3f}", "#e74c3c"), unsafe_allow_html=True)
                with f5:
                    st.markdown(feature_badge("Color H", f"{feats['mean_h']:.1f}", "#9b59b6"), unsafe_allow_html=True)
                with f6:
                    st.markdown(feature_badge("Texture", f"{feats['glcm_contrast']:.2f}", "#1abc9c"), unsafe_allow_html=True)
                
                # Prediction card
                if model is not None:
                    pred_id, pred_name, conf = predict_species(feats, model)
                    
                    st.markdown(f"""
                    <div class="pred-card">
                        <div class="pred-name">{pred_name}</div>
                        <div class="pred-confidence">Confidence: {conf:.1%}</div>
                        <div style="margin-top:10px;background:rgba(255,255,255,0.2);border-radius:5px;height:8px;">
                            <div style="background:#2ecc71;width:{conf*100}%;height:100%;border-radius:5px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Eco traits
                    eco_match = eco_df[eco_df["class_id"] == pred_id]
                    if len(eco_match) == 0:
                        eco_match = eco_df[eco_df["class_id"] == 1]
                    eco_row = eco_match.iloc[0]
                    
                    st.markdown("""
                    <div class="section-card">
                        <div class="section-title">🌿 Ecological Indicators (AVONET)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    e1, e2, e3, e4 = st.columns(4)
                    with e1:
                        st.markdown(eco_card("📏", "Beak", f"{eco_row['beak_length']:.1f}", "mm"), unsafe_allow_html=True)
                    with e2:
                        st.markdown(eco_card("🪽", "Wing", f"{eco_row['wing_length']:.1f}", "mm"), unsafe_allow_html=True)
                    with e3:
                        st.markdown(eco_card("🦶", "Tarsus", f"{eco_row['tarsus_length']:.1f}", "mm"), unsafe_allow_html=True)
                    with e4:
                        st.markdown(eco_card("🪶", "Tail", f"{eco_row['tail_length']:.1f}", "mm"), unsafe_allow_html=True)
            else:
                st.error("❌ Segmentation failed. Please try another image.")

# ==================== PAGE 3: PCA ANALYSIS ====================
elif page == "📈 PCA Analysis":
    st.markdown("<h1 style='color:#1a2f4a;'>PCA Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666;'>Explore bird species clustering</p>", unsafe_allow_html=True)
    
    # Extract family for coloring
    pca_df["family"] = pca_df["class_name"].apply(
        lambda x: x.split(".")[-1].split("_")[-1] if "." in str(x) else "Unknown"
    )
    
    # Top families only for cleaner plot
    top_families = pca_df["family"].value_counts().head(15).index.tolist()
    pca_df["plot_family"] = pca_df["family"].apply(lambda f: f if f in top_families else "Other")
    
    # Plotly scatter
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="plot_family",
        hover_data=["class_name"],
        title="PCA Scatter Plot of Bird Species",
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.7
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        legend=dict(
            title="Family",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=40, r=150, t=60, b=40)
    )
    fig.update_traces(marker=dict(size=6))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Bottom section
    b1, b2 = st.columns([2, 3])
    
    with b1:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">❓ What does this plot mean?</div>
            <p style="color:#555;font-size:14px;line-height:1.6;">
            Each point represents a bird image projected from 10 visual features into 2 dimensions using PCA.
            <br><br>
            Birds with similar morphology, color, and texture patterns cluster together.
            Colors indicate taxonomic family.
            <br><br>
            This helps visualize how classical DIP features capture biological diversity.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with b2:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🔍 10 Nearest Birds (Center)</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Find nearest to center (0,0)
        center_pc1 = pca_df["PC1"].median()
        center_pc2 = pca_df["PC2"].median()
        nearest = find_nearest_birds(center_pc1, center_pc2, n=10)
        
        for _, row in nearest.iterrows():
            img_path = get_image_path(row["filename"])
            cols = st.columns([1, 4, 2])
            with cols[0]:
                if img_path and os.path.exists(img_path):
                    st.image(img_path, width=40)
                else:
                    st.markdown("🐦")
            with cols[1]:
                st.markdown(f"<small><b>{row['class_name']}</b></small>", unsafe_allow_html=True)
            with cols[2]:
                dist = np.sqrt((row["PC1"]-center_pc1)**2 + (row["PC2"]-center_pc2)**2)
                st.markdown(f"<small style='color:#888;'>d={dist:.3f}</small>", unsafe_allow_html=True)
            st.markdown("<hr style='margin:3px 0;opacity:0.2;'>", unsafe_allow_html=True)