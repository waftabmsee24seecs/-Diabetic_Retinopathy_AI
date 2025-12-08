# import os
# import cv2
# import numpy as np
# from PIL import Image
# import streamlit as st
# from tensorflow import keras
# import tensorflow as tf
# from tensorflow.keras import mixed_precision
# from tensorflow.keras.applications.efficientnet import preprocess_input
# import json

# # ==========================================================
# #  MIXED PRECISION (matching training)
# # ==========================================================
# mixed_precision.set_global_policy("mixed_float16")
# print("Mixed precision enabled (float16).")

# # ==========================================================
# #  QWK METRIC (needed for JSON model loading)
# # ==========================================================
# class QWK_Metric(tf.keras.metrics.Metric):
#     def __init__(self, num_classes=5, name="qwk", **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.num_classes = num_classes
#         self.cm = self.add_weight(name="cm", shape=(num_classes, num_classes),
#                                   initializer="zeros", dtype=tf.float32)

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.argmax(y_true, 1)
#         y_pred = tf.argmax(y_pred, 1)
#         m = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32)
#         self.cm.assign_add(m)

#     def result(self):
#         cm = self.cm
#         w = tf.zeros_like(cm)
#         for i in range(self.num_classes):
#             for j in range(self.num_classes):
#                 w = tf.tensor_scatter_nd_update(w, [[i,j]], [(i-j)**2 / (self.num_classes-1)**2])
#         act = tf.reduce_sum(cm, 1)
#         pred = tf.reduce_sum(cm, 0)
#         expected = tf.tensordot(act, pred, axes=0) / tf.reduce_sum(cm)
#         return 1 - tf.reduce_sum(w*cm) / tf.reduce_sum(w*expected)

#     def reset_state(self):
#         self.cm.assign(tf.zeros_like(self.cm))

# # ==========================================================
# #  STREAMLIT CONFIG
# # ==========================================================
# st.set_page_config(
#     page_title="👁️ Diabetic Retinopathy AI Diagnostic Tool",
#     layout="wide"
# )

# # ==========================================================
# #  PATHS
# # ==========================================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# RETINA_MODEL_PATH = os.path.join(BASE_DIR, "retina_model.h5")  # Your model1
# SEVERITY_ARCH_PATH = os.path.join(BASE_DIR, "b4_model_architecture.json")
# SEVERITY_WEIGHTS_PATH = os.path.join(BASE_DIR, "b4_final.weights.h5")

# # ==========================================================
# #  MODEL LOADING
# # ==========================================================
# @st.cache_resource
# def load_retina_model():
#     print("Loading retina vs non-retina model...")
#     model = keras.models.load_model(RETINA_MODEL_PATH)
#     print("Retina model loaded.")
#     return model

# @st.cache_resource
# def load_severity_model():
#     print("Loading DR severity model...")
#     with open(SEVERITY_ARCH_PATH, "r") as f:
#         json_arch = f.read()
#     model = keras.models.model_from_json(json_arch, custom_objects={"QWK_Metric": QWK_Metric})
#     model.load_weights(SEVERITY_WEIGHTS_PATH)
#     print("Severity model loaded.")
#     return model

# retina_model = load_retina_model()
# severity_model = load_severity_model()

# # ==========================================================
# #  IMAGE PREPROCESSING
# # ==========================================================
# IMG_SIZE = (300, 300)

# def preprocess_image_retina(img):
#     img = img.resize(IMG_SIZE)
#     img = np.array(img).astype(np.float32)
#     img = np.expand_dims(img, axis=0)
#     return img

# def preprocess_image_severity(img):
#     img = img.resize(IMG_SIZE)
#     img = np.array(img).astype(np.float32)
#     img = preprocess_input(img)
#     img = np.expand_dims(img, axis=0)
#     return img

# # ==========================================================
# #  PREDICTION FUNCTIONS
# # ==========================================================
# def predict_retina(img):
#     x = preprocess_image_retina(img)
#     pred = retina_model.predict(x)
#     label = ["non-retinal", "retinal"][int(np.argmax(pred))]
#     confidence = float(np.max(pred))
#     return label, confidence

# def predict_severity(img):
#     x = preprocess_image_severity(img)
#     pred = severity_model.predict(x)
#     classes = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
#     label = classes[int(np.argmax(pred))]
#     confidence = float(np.max(pred))
#     return label, confidence

# # ==========================================================
# #  STREAMLIT UI & CSS
# # ==========================================================
# st.markdown("""
# <style>
# /* Add your File2 custom CSS here (for brevity omitted, keep your existing CSS) */
# </style>
# """, unsafe_allow_html=True)

# st.markdown("<div class='stCard'>", unsafe_allow_html=True)
# st.title("👁️ Diabetic Retinopathy AI Diagnostic Tool")
# st.markdown("Upload a fundus image to analyze retina presence and DR severity.")
# st.markdown("</div>", unsafe_allow_html=True)

# # ==========================================================
# #  IMAGE UPLOAD OR CAMERA INPUT
# # ==========================================================
# if 'input_mode' not in st.session_state:
#     st.session_state.input_mode = None

# image_input = None

# if st.session_state.input_mode is None:
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("⬆️ Upload an Image File"):
#             st.session_state.input_mode = "file"
#             st.stop()  # replaces experimental_rerun
#     with col2:
#         if st.button("📸 Capture with Camera"):
#             st.session_state.input_mode = "camera"
#             st.stop()  # replaces experimental_rerun

# elif st.session_state.input_mode == "file":
#     col_input, col_reset = st.columns([4,1])
#     with col_input:
#         image_input = st.file_uploader("Choose an image file...", type=["jpg","jpeg","png"])
#     with col_reset:
#         if st.button("Change Input"):
#             st.session_state.input_mode = None
#             st.stop()  # replaces experimental_rerun

# elif st.session_state.input_mode == "camera":
#     col_input, col_reset = st.columns([4,1])
#     with col_input:
#         image_input = st.camera_input("Take a photo of the fundus image")
#     with col_reset:
#         if st.button("Change Input"):
#             st.session_state.input_mode = None
#             st.stop()  # replaces experimental_rerun

# # ==========================================================
# #  ANALYSIS & RESULTS
# # ==========================================================
# if image_input is not None:
#     image = Image.open(image_input).convert("RGB")

#     col_image, col_retina_result, col_severity_result = st.columns(3)

#     with col_image:
#         st.markdown("<div class='image-display-card'>", unsafe_allow_html=True)
#         st.subheader("1. Input Image")
#         st.image(image, use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)

#     with col_retina_result:
#         st.markdown("<div class='result-card'>", unsafe_allow_html=True)
#         st.subheader("2. Retina Check")
#         label, conf = predict_retina(image)
#         st.markdown(f"**Classification:** {label.upper()}")
#         st.metric("Confidence", f"{conf*100:.2f}%")
#         if label.lower() == "retinal":
#             st.success("✅ Image is suitable for DR screening.")
#         else:
#             st.error("❌ Non-retinal image. DR screening aborted.")
#         st.markdown("</div>", unsafe_allow_html=True)

#     with col_severity_result:
#         st.markdown("<div class='result-card'>", unsafe_allow_html=True)
#         st.subheader("3. DR Severity")
#         if label.lower() == "retinal":
#             sev_label, sev_conf = predict_severity(image)
#             css_class = sev_label.replace(" ","").replace("-","")
#             st.markdown(f"<p class='severity-level {css_class}'><strong>{sev_label.upper()}</strong></p>", unsafe_allow_html=True)
#             st.metric("Prediction Confidence", f"{sev_conf*100:.2f}%")
#             st.success("✅ Full diagnosis complete.")
#         else:
#             st.info("Awaiting valid retinal image.")
#         st.markdown("</div>", unsafe_allow_html=True)

# else:
#     st.info("⬆️ Select an input method above to start the diagnostic process.")


#!/usr/bin/env python3
# dr_app_streamlit.py
# Single-file Streamlit app: local inference (no Flask), image saving, camera support,
# enhanced visual output cards with colors and icons. Drop-in replacement for vis.py UI.
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import io
import uuid
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.efficientnet import preprocess_input

# --------------------------
# Mixed precision
# --------------------------
mixed_precision.set_global_policy("mixed_float16")
print("Mixed precision enabled (float16).")

# --------------------------
# QWK Metric (same as training)
# --------------------------
class QWK_Metric(tf.keras.metrics.Metric):
    def __init__(self, num_classes=5, name="qwk", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.cm = self.add_weight(
            name="cm",
            shape=(num_classes, num_classes),
            initializer="zeros",
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, 1)
        y_pred = tf.argmax(y_pred, 1)
        m = tf.math.confusion_matrix(
            y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32
        )
        self.cm.assign_add(m)

    def result(self):
        cm = self.cm
        w = tf.zeros_like(cm)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                w = tf.tensor_scatter_nd_update(
                    w,
                    [[i, j]],
                    [(i - j) ** 2 / (self.num_classes - 1) ** 2]
                )
        act = tf.reduce_sum(cm, 1)
        pred = tf.reduce_sum(cm, 0)
        expected = tf.tensordot(act, pred, axes=0) / tf.reduce_sum(cm)
        return 1 - tf.reduce_sum(w * cm) / tf.reduce_sum(w * expected)

    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))

# --------------------------
# Helpers
# --------------------------
def bytes_to_cv2_image(image_bytes):
    """Convert bytes to OpenCV BGR image or return None."""
    try:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def save_uploaded_image(image_bytes, original_filename=None):
    """Save uploaded or camera-captured image to uploadedimages/ and return path."""
    if not os.path.exists("uploadedimages"):
        os.makedirs("uploadedimages")

    if original_filename and isinstance(original_filename, str) and original_filename.strip() != "":
        filename = original_filename
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"camera_capture_{timestamp}_{unique_id}.png"

    save_path = os.path.join("uploadedimages", filename)

    # ensure bytes
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    return save_path

# --------------------------
# Model loading (cached)
# --------------------------
@st.cache_resource
def load_retina_model():
    path = "retina_model_final.h5"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found in current directory.")
    return keras.models.load_model(path, compile=False)

@st.cache_resource
def load_severity_model():
    arch_path = "b4_model_architecture.json"
    weights_path = "b4_final.weights.h5"
    if not os.path.exists(arch_path):
        raise FileNotFoundError(f"{arch_path} not found in current directory.")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"{weights_path} not found in current directory.")
    with open(arch_path, "r") as f:
        json_arch = f.read()
    model = keras.models.model_from_json(json_arch, custom_objects={"QWK_Metric": QWK_Metric})
    model.load_weights(weights_path)
    return model

# --------------------------
# Prediction functions
# --------------------------
def predict_retina_local(image_bytes):
    labels = ['non-retinal', 'retinal']
    IMG_SIZE = (300, 300)

    img = bytes_to_cv2_image(image_bytes)
    if img is None:
        return {"message": "non-retinal", "confidence": 0.0}

    try:
        img = cv2.resize(img, IMG_SIZE)
    except Exception:
        return {"message": "non-retinal", "confidence": 0.0}

    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    model = load_retina_model()
    pred = model.predict(img, verbose=0)
    conf = float(np.max(pred))
    label = labels[int(np.argmax(pred))]
    return {"message": label, "confidence": conf}

def predict_severity_local(image_bytes):
    labels = [
        'No DR',
        'Mild DR',
        'Moderate DR',
        'Severe DR',
        'Proliferative DR'
    ]
    IMG_SIZE = (300, 300)

    img = bytes_to_cv2_image(image_bytes)
    if img is None:
        return {"message": "No DR", "confidence": 0.0}

    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
    except Exception:
        return {"message": "No DR", "confidence": 0.0}

    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    model = load_severity_model()
    pred = model.predict(img, verbose=0)
    conf = float(np.max(pred))
    label = labels[int(np.argmax(pred))]
    return {"message": label, "confidence": conf}

# --------------------------
# Replacement API-like functions (GUI unchanged)
# --------------------------
def call_retinal_classifier(image_file):
    """
    image_file: bytes or file-like (UploadedFile / BytesIO)
    returns {"message":..., "confidence":...}
    """
    try:
        if isinstance(image_file, (bytes, bytearray)):
            b = image_file
        else:
            try:
                image_file.seek(0)
            except Exception:
                pass
            b = image_file.read()

        return predict_retina_local(b)
    except Exception as e:
        st.error(f"Retina classifier error: {e}")
        return {"message": "Unknown API Error", "confidence": 0.0}

def call_severity_classifier(image_file):
    try:
        if isinstance(image_file, (bytes, bytearray)):
            b = image_file
        else:
            try:
                image_file.seek(0)
            except Exception:
                pass
            b = image_file.read()

        return predict_severity_local(b)
    except Exception as e:
        st.error(f"Severity classifier error: {e}")
        return {"message": "Unknown API Error", "confidence": 0.0}

# --------------------------
# Streamlit UI (GUI kept same, styles extended)
# --------------------------
API_BASE_URL = "http://localhost:5000"  # kept for GUI parity

st.set_page_config(page_title="Diabetic Retinopathy AI Diagnostic", layout="wide", initial_sidebar_state="expanded")

if 'input_mode' not in st.session_state:
    st.session_state.input_mode = None

# Enhanced CSS (keeps original layout tuning)
custom_css = """
<style>
/* Keep original layout tuning */
#stApp { padding-top: 0px !important; margin-top: 0px !important; }
section.main { padding-top: 0px !important; margin-top: 0px !important; }
.stApp > header { height: 0px !important; visibility: hidden; }
body { font-family: 'Inter', sans-serif; color: #333; background-color: #f0f2f6; }
/* Card styling - These are used ONLY when an image is uploaded for the results */
.stCard, .result-card, .input-card, .image-display-card {
    background-color: #ffffff;
    border-radius: 12px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    padding: 18px;
    margin-bottom: 20px;
    border: 1px solid #e8eaef;
}
/* IMPORTANT FIX: Enforce a minimum height for the display card to prevent layout shifting */
.image-display-card { min-height: 360px; display:flex; flex-direction:column; align-items:center; justify-content:flex-start; } 
/* Result card small text */
.result-card h3 { color:#1a73e8 !important; }
/* Retina result styling placeholder */
.result-badge {
    padding: 16px;
    border-radius: 12px;
    display:block;
    width:100%;
}
.severity-level {
    padding: 16px;
    border-radius: 12px;
    display:block;
    width:100%;
}
/* Confidence badge */
.conf-badge {
    display:inline-block;
    padding:8px 12px;
    border-radius: 999px;
    font-weight:600;
    font-size:1rem;
    margin-top:8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
/* Icons size */
.result-icon { font-size:1.6rem; margin-right:8px; vertical-align:middle; }
/* small responsive tweaks */
@media (max-width:900px){
  .image-display-card { min-height:220px; }
}

/* Ensure subheader in the main flow is styled like others and remove top margin */
h3 { margin-top: 0px !important; }

/* FIX 1: Aggressively target and reduce margins/padding on the input widgets themselves */
.stFileUploader, .stCameraInput {
    margin-bottom: 0px !important;
    padding-bottom: 0px !important;
}
/* FIX 2: Attempt to counter the file uploader height change (existing fix, keep it) */
.stFileUploader > div:first-child, .stCameraInput > div:first-child {
    margin-bottom: -15px !important; 
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Title card (unchanged)
st.markdown("<div class='stCard'>", unsafe_allow_html=True)
st.title("👁️ Diabetic Retinopathy AI Diagnostic Tool")
st.markdown("A demonstration interface for deep learning models used in ophthalmology.")
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar content unchanged
with st.sidebar:
    st.header("About This Demo")
    st.info("This application demonstrates a two-stage process for diagnosing Diabetic Retinopathy (DR): first, filtering non-retinal images, and second, classifying the severity level.")
    st.markdown(f"**API Endpoint:** `{API_BASE_URL}`")
    st.markdown("---")
    st.subheader("Classification Models")
    st.markdown("- **Model 1 (Stage 1):** Retina vs. Non-Retina\n- **Model 2 (Stage 2):** DR Severity (No DR, Mild, Moderate, Severe, Proliferative)")
    st.markdown("---")
    st.markdown("Developed using Streamlit and TensorFlow. Models are loaded locally.")

# Input selection UI 
st.markdown("<div class='input-card'>", unsafe_allow_html=True)
st.subheader("Select Image Input Method")

image_input = None

# Using a container for the input widgets helps control their vertical space
input_container = st.container()

if st.session_state.input_mode is None:
    with input_container:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬆️ Upload an Image File", use_container_width=True):
                st.session_state.input_mode = "file"
                st.rerun()
        with col2:
            if st.button("📸 Capture with Camera", use_container_width=True):
                st.session_state.input_mode = "camera"
                st.rerun()

elif st.session_state.input_mode == "file":
    with input_container:
        # FIX: Using st.markdown instead of st.info to reduce vertical space
        st.markdown("**Upload Mode Selected.** Click 'Change Input' to switch to camera.")
        col_input, col_reset = st.columns([4,1])
        with col_input:
            image_input = st.file_uploader("Choose an image file...", type=["jpg","jpeg","png"])
        with col_reset:
            # FIX: Using st.write("") instead of <br> to add minimal necessary vertical space
            st.write("") 
            if st.button("Change Input", key="reset_file", use_container_width=True):
                st.session_state.input_mode = None
                st.rerun()

elif st.session_state.input_mode == "camera":
    with input_container:
        # FIX: Using st.markdown instead of st.info to reduce vertical space
        st.markdown("**Camera Mode Selected.** Click 'Change Input' to switch to file upload.")
        col_input, col_reset = st.columns([4,1])
        with col_input:
            image_input = st.camera_input("Take a photo of the fundus image")
        with col_reset:
            # FIX: Using st.write("") instead of <br> to add minimal necessary vertical space
            st.write("") 
            if st.button("Change Input", key="reset_camera", use_container_width=True):
                st.session_state.input_mode = None
                st.rerun()
st.markdown("</div>", unsafe_allow_html=True) # End of the input-card div

# --- Define Columns ONCE before the if/else block ---
# Use a consistent set of names for the columns
col_image, col_retina_result, col_severity_result = st.columns(3)


# --- Logic for displaying image and results ---

if image_input is not None:
    # Logic for when image IS uploaded (Active Results)

    # 1. Processing
    try:
        image_input.seek(0)
    except Exception:
        pass
    image_bytes = image_input.read()

    try:
        if hasattr(image_input, "name") and image_input.name:
            saved_path = save_uploaded_image(image_bytes, image_input.name)
        else:
            saved_path = save_uploaded_image(image_bytes, None)
    except Exception as e:
        st.warning(f"⚠ Could not save image: {e}")

    pil_img = Image.open(io.BytesIO(image_bytes))

    # 2. Image Display (Col 1)
    with col_image:
        st.subheader("1. Input Image")
        st.markdown("<div class='image-display-card'>", unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        st.caption(f"Saved to: {saved_path}")
        st.markdown("</div>", unsafe_allow_html=True) 

    # 3. Retina Check (Col 2)
    with st.spinner("Analyzing image with AI models..."):
        retinal_check_result = call_retinal_classifier(image_bytes)
        result_label = retinal_check_result.get("message", "N/A")
        confidence_score = retinal_check_result.get("confidence", 0.0)

    with col_retina_result:
        st.subheader("2. Retina Check (Stage 1)")
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        if result_label.lower() in ["retinal", "retina", "retinal image"]:
            bg = "#e6ffe6"
            border = "#1b5e20"
            icon = "✅"
            title = "RETINAL"
            st_message = st.success
        else:
            bg = "#fff5f5"
            border = "#b71c1c"
            icon = "❌"
            title = result_label.upper()
            st_message = st.error

        st.markdown(f"""
            <div class="result-badge" style="background:{bg}; border-left:6px solid {border};">
                <span class="result-icon">{icon}</span>
                <span style="font-size:1.5rem; font-weight:700; color:{border}; margin-left:6px;">{title}</span>
                <div style="margin-top:8px;">
                    <span class="conf-badge" style="background:white; color:{border};">Confidence: {confidence_score*100:.2f}%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if result_label.lower() == 'retinal':
            st_message("✅ Image is valid for DR screening.")
        else:
            st_message("❌ Image is non-retinal. DR screening aborted.")

        st.markdown("</div>", unsafe_allow_html=True)

    # 4. Severity (Col 3)
    with col_severity_result:
        st.subheader("3. DR Severity (Stage 2)")
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        if result_label.lower() == 'retinal':
            with st.spinner("Running DR severity model..."):
                severity_result = call_severity_classifier(image_bytes)
                severity_label = severity_result.get("message", "N/A")
                severity_confidence = severity_result.get("confidence", 0.0)

            severity_colors = {
                "No DR": ("#e6ffe6", "#1b5e20", "✅"),
                "Mild DR": ("#fff8e1", "#ff8f00", "⚠️"),
                "Moderate DR": ("#fff3e0", "#e65100", "⚠️"),
                "Severe DR": ("#ffebee", "#b71c1c", "🔥"),
                "Proliferative DR": ("#f3e5f5", "#6a1b9a", "⚠️")
            }
            bg, border, icon = severity_colors.get(severity_label, ("#f0f0f0", "#333", "ℹ️"))

            st.markdown(f"""
                <div class="severity-level" style="background:{bg}; border-left:6px solid {border};">
                    <span class="result-icon">{icon}</span>
                    <span style="font-size:1.4rem; font-weight:700; color:{border}; margin-left:6px;">{severity_label.upper()}</span>
                    <div style="margin-top:8px;">
                        <span class="conf-badge" style="background:white; color:{border};">Confidence: {severity_confidence*100:.2f}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.success("✅ Full diagnosis complete.")

        else:
            st.info("Awaiting valid retinal image from Stage 1.")

        st.markdown("</div>", unsafe_allow_html=True)

else:
    # Logic for when NO image is uploaded (Placeholders)

    # Placeholder 1: Input Image
    with col_image:
        st.subheader("1. Input Image")
        # FIX: Use the same wrapper (image-display-card) for consistent styling
        st.markdown("<div class='image-display-card' style='padding: 0;'>", unsafe_allow_html=True) 
        st.markdown(
            """
            <div style='background-color: #f0f2f6; color: #6c757d; padding: 60px 10px; border-radius: 12px; text-align: center; font-weight: 600; font-size: 1.2rem; min-height: 250px; display: flex; align-items: center; justify-content: center; height: 100%; border: 2px dashed #ced4da;'>
                <p style='margin: 0;'>Upload Fundus Image To Start</p>
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True) # Close the wrapper
        st.caption("Upload or capture an image above to begin analysis.")


    # Placeholder 2: Retina Check
    with col_retina_result:
        st.subheader("2. Retina Check (Stage 1)")
        # FIX: Use the result-card wrapper for consistent height/look
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.info("Awaiting image upload to start analysis.")
        st.markdown("</div>", unsafe_allow_html=True) 

    # Placeholder 3: DR Severity
    with col_severity_result:
        st.subheader("3. DR Severity (Stage 2)")
        # FIX: Use the result-card wrapper for consistent height/look
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.info("Awaiting valid result from Stage 1.")
        st.markdown("</div>", unsafe_allow_html=True) 

st.markdown("---")
st.caption("Disclaimer: This tool is for demonstration purposes only and is not a substitute for professional medical advice.")
