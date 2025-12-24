import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="TB Detection App",
    page_icon="🫁",
    layout="wide"
)

# =============================
# CUSTOM CSS
# =============================
st.markdown("""
<style>
[data-testid="stSidebar"] { font-size: 18px !important; }
.main-title { font-size: 36px; font-weight: bold; color: #2C3E50; }
.stMarkdown, .stText, .stAlert { font-size: 18px; }
</style>
""", unsafe_allow_html=True)

# =============================
# MODEL PATHS (MATCH YOUR REPO)
# =============================
MODEL_PATHS = {
    "ResNet50": "tb_model.keras",
    "VGG16": "vgg16_tb_model.h5",
    "EfficientNetB0": "efficientnet_tb.h5"
}

IMG_SIZE = (224, 224)

# =============================
# LOAD MODEL (CACHED)
# =============================
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# =============================
# SIDEBAR
# =============================
st.sidebar.title("⚙️ Application Menu")
page = st.sidebar.radio("Navigate", ["Home", "Model Info", "Prediction"])

# =============================
# HOME PAGE
# =============================
if page == "Home":
    st.markdown('<p class="main-title">🫁 Tuberculosis Detection using Deep Learning</p>', unsafe_allow_html=True)

    st.write("""
    This application detects **Tuberculosis (TB)** from **Chest X-ray images**
    using **Deep Learning models**.
    """)

    st.subheader("📌 About Project")
    st.write("""
    The system classifies chest X-rays into:
    - 🟢 Normal
    - 🔴 Tuberculosis Detected
    """)

    st.subheader("✨ Key Features")
    st.markdown("""
    ✅ Multiple CNN models  
    ✅ High accuracy & F1-score  
    ✅ Simple and clean UI  
    ✅ Instant prediction  
    """)

    st.info("⚠️ Educational use only. Not a medical diagnosis tool.")

# =============================
# MODEL INFO PAGE
# =============================
elif page == "Model Info":
    st.header("📊 Model Comparison")

    df = pd.DataFrame({
        "Model": ["ResNet50", "VGG16", "EfficientNetB0"],
        "Accuracy (%)": [92.5, 89.7, 93.1],
        "F1-Score": [0.92, 0.88, 0.93]
    })

    st.table(df)

    best = df.loc[df["F1-Score"].idxmax()]
    st.success(f"🏆 Best Model: **{best['Model']}**")

# =============================
# PREDICTION PAGE
# =============================
elif page == "Prediction":
    st.markdown('<p class="main-title">🩻 TB Prediction</p>', unsafe_allow_html=True)

    model_choice = st.sidebar.selectbox("Select Model", MODEL_PATHS.keys())
    model = load_model(MODEL_PATHS[model_choice])

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        img = image.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        confidence = round(float(prediction if prediction >= 0.5 else 1 - prediction) * 100, 2)

        if prediction >= 0.5:
            st.error(f"⚠️ TB Detected — Confidence: {confidence}%")
        else:
            st.success(f"✅ Normal — Confidence: {confidence}%")
