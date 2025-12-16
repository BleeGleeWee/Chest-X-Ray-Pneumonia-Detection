import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

# Handle import error in case gradcam.py is missing locally
try:
    from gradcam import get_gradcam_heatmap
except ImportError:
    def get_gradcam_heatmap(model, img_array, layer_name):
        return np.zeros((224, 224))

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

# ----------------- CSS: BLOODSTREAM VISUALS -----------------
st.markdown(f"""
<style>
/* 1. GLOBAL RESET & SCROLLBAR HIDING */
* {{ user-select: none; }}
::-webkit-scrollbar {{ display: none; }}

/* 2. BACKGROUND: DEEP BLOODSTREAM GRADIENT */
.stApp {{
    /* Deep red to black gradient simulating depth */
    background: radial-gradient(circle at 50% 50%, #4a0000 0%, #2a0000 40%, #000000 100%);
    background-attachment: fixed;
    background-size: cover;
    overflow: hidden; /* Prevent scrollbars from moving elements */
}}

/* 3. REALISTIC RED BLOOD CELL (CSS ONLY) */
.blood-cell {{
    position: fixed;
    border-radius: 50%;
    /* The base red color with a highlight */
    background: radial-gradient(circle at 30% 30%, #ff5e5e, #800000);
    /* The "Inset" shadow creates the biconcave (donut) dip in the center */
    box-shadow: 
        inset 10px 10px 20px rgba(0,0,0,0.5), /* Deep dark center */
        inset -5px -5px 15px rgba(255,255,255,0.2), /* Subtle rim reflection */
        5px 5px 15px rgba(0,0,0,0.3); /* Drop shadow for depth */
    opacity: 0.8;
    z-index: 0; /* Behind the content */
    pointer-events: none; /* Allow clicking through them */
    animation: rushDiagonal linear infinite;
}}

/* 4. DIAGONAL RUSH ANIMATION */
@keyframes rushDiagonal {{
    0% {{
        transform: translate(-100px, -100px) rotate(0deg) scale(0.8);
        opacity: 0;
    }}
    10% {{ opacity: 0.8; }}
    90% {{ opacity: 0.8; }}
    100% {{
        /* Move way past the bottom right corner */
        transform: translate(120vw, 120vh) rotate(360deg) scale(1.2);
        opacity: 0;
    }}
}}

/* 5. GLASSMORPHISM CONTAINER */
.block-container {{
    position: relative;
    z-index: 2; /* Sits above the blood cells */
    background: rgba(0, 0, 0, 0.6); /* Darker glass for better contrast */
    backdrop-filter: blur(8px); /* Blur the cells passing behind */
    border-radius: 20px;
    padding: 3rem;
    border: 1px solid rgba(139, 0, 0, 0.3); /* Subtle red border */
    max_width: 800px;
    margin: auto;
}}

/* 6. TEXT & WIDGET STYLING */
h1, h2, h3, h4, h5, h6, label, .stMarkdown p, .stText, .stFileUploader label {{ 
    color: #ffebee !important; 
    text-shadow: 0 2px 4px rgba(0,0,0,0.8);
}}

/* Input/Widget text adjustments */
.stFileUploader div {{ color: white !important; }}
div[data-baseweb="file-uploader"] {{
    border: 1px dashed #ff4d4d;
}}

/* Buttons */
button {{
    background-color: rgba(139, 0, 0, 0.4) !important;
    color: white !important;
    border: 1px solid #ff4d4d !important;
    transition: all 0.3s;
}}
button:hover {{
    background-color: #ff4d4d !important;
    color: black !important;
    box-shadow: 0 0 15px #ff4d4d;
}}

footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ----------------- PYTHON LOGIC: GENERATE FLOATING CELLS -----------------
# We generate random parameters for 20 blood cells so they look organic
# "Rushing" feel comes from faster durations (e.g., 5s to 12s)
html_cells = ""
for i in range(20):
    # Random properties
    left_pos = random.randint(-10, 90)  # Start slightly off-screen to mid-screen
    top_pos = random.randint(-10, 20)  # Start mostly at top
    size = random.randint(40, 120)  # Varying size (depth perception)
    duration = random.uniform(5, 12)  # Varying speeds (fast rush)
    delay = random.uniform(0, 10)  # Don't all start at once

    html_cells += f"""
    <div class="blood-cell" style="
        left: {left_pos}%; 
        top: {top_pos}%; 
        width: {size}px; 
        height: {size}px; 
        animation-duration: {duration}s; 
        animation-delay: -{delay}s;">
    </div>
    """

st.markdown(html_cells, unsafe_allow_html=True)

# ------------------ MAIN APP CONTENT ------------------

col1, col2 = st.columns([1, 8])
with col1:
    st.write("")  # Spacer
with col2:
    st.title("🫁 Chest X-Ray Pneumonia Detection")
    st.markdown("### 🧬 AI-Powered Diagnostics")

# ------------------ DISCLAIMER ------------------
with st.expander("⚠️ Medical Disclaimer"):
    st.warning(
        """
        This application is for **educational purposes only**.
        It is **NOT a medical diagnostic tool**.
        Always consult a **certified medical professional**.
        """
    )


# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_model.keras")


try:
    model = load_model()
except OSError:
    st.error("🚨 Model file not found. Please ensure 'pneumonia_model.keras' is in the directory.")
    st.stop()

LAST_CONV_LAYER = "conv5_block16_concat"

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

show_gradcam = st.toggle("🔥 Show Grad-CAM Heatmap")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_resized = image.resize((224, 224))

    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ------------------ PREDICTION ------------------
    prediction = model.predict(img_array)[0][0]

    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Dynamic neon color for result text
    color_code = "#ff3333" if prediction > 0.5 else "#33ff33"  # Neon Red vs Neon Green
    st.markdown(f"""
    <div style="background: rgba(0,0,0,0.5); padding: 20px; border-radius: 15px; border: 1px solid {color_code}; text-align: center;">
        <h2 style="margin:0; color: white;">Result: <span style='color:{color_code}; font-weight: bold;'>{label}</span></h2>
    </div>
    """, unsafe_allow_html=True)

    st.write("")  # Spacer

    # ------------------ CONFIDENCE BAR ------------------
    st.markdown("### 📊 Model Confidence")
    st.progress(float(confidence))
    st.write(f"Confidence: **{confidence * 100:.2f}%**")

    # ------------------ GRAD-CAM ------------------
    if show_gradcam:
        heatmap = get_gradcam_heatmap(
            model=model,
            img_array=img_array,
            layer_name=LAST_CONV_LAYER
        )

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        # Transparent background for the plot
        fig.patch.set_alpha(0.0)

        ax[0].imshow(image)
        ax[0].set_title("Original X-ray", color='white')
        ax[0].axis("off")

        ax[1].imshow(image)
        ax[1].imshow(heatmap, cmap="jet", alpha=0.4)
        ax[1].set_title("Grad-CAM Heatmap", color='white')
        ax[1].axis("off")

        st.pyplot(fig)

    else:
        st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)

st.markdown("---")
st.caption("🔓 Free public Streamlit Cloud deployment ready")