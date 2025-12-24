import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# =============================
# 🔹 PAGE CONFIG
# =============================
st.set_page_config(
    page_title="TB Detection App",
    page_icon="🫁",
    layout="wide"
)

# =============================
# 🔹 Custom CSS
# =============================
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            font-size: 18px !important;
        }
        .main-title {
            font-size: 36px !important;
            font-weight: bold;
            color: #2C3E50;
        }
        .sub-title {
            font-size: 22px !important;
            font-weight: bold;
            color: #34495E;
        }
        .stMarkdown, .stText, .stAlert {
            font-size: 18px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# 🔹 MODEL PATHS (CLOUD SAFE)
# =============================
MODEL_PATHS = {
    "ResNet50": "models/tb_model.keras",
    "VGG16": "models/vgg16_tb_model.h5",
    "EfficientNetB0": "models/efficientnet_tb_model.h5",
}

IMG_SIZE = (224, 224)

# =============================
# 🔹 LOAD MODEL (CACHED)
# =============================
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# =============================
# 🔹 SIDEBAR
# =============================
st.sidebar.title("⚙️ Application Menu")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Model Info", "Prediction"]
)

# =============================
# 🔹 HOME PAGE
# =============================
if page == "Home":
    st.markdown('<p class="main-title">🫁 Tuberculosis Detection using Deep Learning</p>', unsafe_allow_html=True)

    st.markdown("""
    Welcome to the **Tuberculosis Detection App**.  
    This application detects **Tuberculosis (TB)** from **Chest X-ray images**
    using advanced **Deep Learning models**.
    """)

    st.image(
        "assets/sample_xray.png",
        caption="Chest X-ray Example",
        use_container_width=True
    )

    st.subheader("📌 About the Project")
    st.write("""
    Tuberculosis is a serious infectious disease affecting millions worldwide.  
    This system uses **CNN-based deep learning models** to classify X-rays into:
    - 🟢 Normal  
    - 🔴 Tuberculosis Detected
    """)

    st.subheader("✨ Key Features")
    st.markdown("""
    ✅ Multiple Deep Learning Models  
    ✅ High Accuracy & F1-Score  
    ✅ User-friendly Streamlit UI  
    ✅ Instant Prediction  
    """)

    st.subheader("🛠️ How to Use")
    st.markdown("""
    1️⃣ Go to **Prediction** page  
    2️⃣ Upload Chest X-ray image  
    3️⃣ Select a model  
    4️⃣ View prediction & confidence  
    """)

    st.info("⚠️ Educational purpose only. Not a medical diagnosis tool.")

# =============================
# 🔹 MODEL INFO PAGE
# =============================
elif page == "Model Info":
    st.header("📊 Model Information & Comparison")

    st.subheader("🧠 Architectures Used")
    st.write("""
    - **ResNet50** – Deep residual learning  
    - **VGG16** – Classical CNN with strong performance  
    - **EfficientNetB0** – Lightweight & efficient  
    """)

    model_comparison = pd.DataFrame({
        "Model": ["ResNet50", "VGG16", "EfficientNetB0"],
        "Parameters (Millions)": [25.6, 138, 5.3],
        "Training Time (hrs)": ["~2.5", "~3.5", "~1.5"],
        "Accuracy (%)": [92.5, 89.7, 93.1],
        "F1-Score": [0.92, 0.88, 0.93]
    })

    st.table(model_comparison)

    best_model = model_comparison.loc[model_comparison["F1-Score"].idxmax()]
    st.success(
        f"🏆 Best Model: **{best_model['Model']}** (F1-Score = {best_model['F1-Score']})"
    )

# =============================
# 🔹 PREDICTION PAGE
# =============================
elif page == "Prediction":
    st.markdown('<p class="main-title">🩻 TB Prediction from X-ray</p>', unsafe_allow_html=True)

    model_choice = st.sidebar.selectbox(
        "Select Model",
        list(MODEL_PATHS.keys())
    )

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

        confidence = round(
            float(prediction if prediction >= 0.5 else 1 - prediction) * 100, 2
        )

        if prediction >= 0.5:
            st.error(f"⚠️ **TB Detected** — Confidence: {confidence}%")
        else:
            st.success(f"✅ **Normal** — Confidence: {confidence}%")
