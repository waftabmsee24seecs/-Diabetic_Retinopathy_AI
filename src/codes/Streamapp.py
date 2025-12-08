import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.efficientnet import preprocess_input
import json

# ==========================================================
#  MIXED PRECISION (matching training)
# ==========================================================
mixed_precision.set_global_policy("mixed_float16")
print("Mixed precision enabled (float16).")

# ==========================================================
#  QWK METRIC (needed for JSON model loading)
# ==========================================================
class QWK_Metric(tf.keras.metrics.Metric):
    def __init__(self, num_classes=5, name="qwk", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.cm = self.add_weight(name="cm", shape=(num_classes, num_classes),
                                  initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, 1)
        y_pred = tf.argmax(y_pred, 1)
        m = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32)
        self.cm.assign_add(m)

    def result(self):
        cm = self.cm
        w = tf.zeros_like(cm)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                w = tf.tensor_scatter_nd_update(w, [[i,j]], [(i-j)**2 / (self.num_classes-1)**2])
        act = tf.reduce_sum(cm, 1)
        pred = tf.reduce_sum(cm, 0)
        expected = tf.tensordot(act, pred, axes=0) / tf.reduce_sum(cm)
        return 1 - tf.reduce_sum(w*cm) / tf.reduce_sum(w*expected)

    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))

# ==========================================================
#  STREAMLIT CONFIG
# ==========================================================
st.set_page_config(
    page_title="👁️ Diabetic Retinopathy AI Diagnostic Tool",
    layout="wide"
)

# ==========================================================
#  PATHS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RETINA_MODEL_PATH = os.path.join(BASE_DIR, "retina_model.h5")  # Your model1
SEVERITY_ARCH_PATH = os.path.join(BASE_DIR, "b4_model_architecture.json")
SEVERITY_WEIGHTS_PATH = os.path.join(BASE_DIR, "b4_final.weights.h5")

# ==========================================================
#  MODEL LOADING
# ==========================================================
@st.cache_resource
def load_retina_model():
    print("Loading retina vs non-retina model...")
    model = keras.models.load_model(RETINA_MODEL_PATH)
    print("Retina model loaded.")
    return model

@st.cache_resource
def load_severity_model():
    print("Loading DR severity model...")
    with open(SEVERITY_ARCH_PATH, "r") as f:
        json_arch = f.read()
    model = keras.models.model_from_json(json_arch, custom_objects={"QWK_Metric": QWK_Metric})
    model.load_weights(SEVERITY_WEIGHTS_PATH)
    print("Severity model loaded.")
    return model

retina_model = load_retina_model()
severity_model = load_severity_model()

# ==========================================================
#  IMAGE PREPROCESSING
# ==========================================================
IMG_SIZE = (300, 300)

def preprocess_image_retina(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_image_severity(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img).astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# ==========================================================
#  PREDICTION FUNCTIONS
# ==========================================================
def predict_retina(img):
    x = preprocess_image_retina(img)
    pred = retina_model.predict(x)
    label = ["non-retinal", "retinal"][int(np.argmax(pred))]
    confidence = float(np.max(pred))
    return label, confidence

def predict_severity(img):
    x = preprocess_image_severity(img)
    pred = severity_model.predict(x)
    classes = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
    label = classes[int(np.argmax(pred))]
    confidence = float(np.max(pred))
    return label, confidence

# ==========================================================
#  STREAMLIT UI & CSS
# ==========================================================
st.markdown("""
<style>
/* Add your File2 custom CSS here (for brevity omitted, keep your existing CSS) */
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='stCard'>", unsafe_allow_html=True)
st.title("👁️ Diabetic Retinopathy AI Diagnostic Tool")
st.markdown("Upload a fundus image to analyze retina presence and DR severity.")
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
#  IMAGE UPLOAD OR CAMERA INPUT
# ==========================================================
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = None

image_input = None

if st.session_state.input_mode is None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬆️ Upload an Image File"):
            st.session_state.input_mode = "file"
            st.stop()  # replaces experimental_rerun
    with col2:
        if st.button("📸 Capture with Camera"):
            st.session_state.input_mode = "camera"
            st.stop()  # replaces experimental_rerun

elif st.session_state.input_mode == "file":
    col_input, col_reset = st.columns([4,1])
    with col_input:
        image_input = st.file_uploader("Choose an image file...", type=["jpg","jpeg","png"])
    with col_reset:
        if st.button("Change Input"):
            st.session_state.input_mode = None
            st.stop()  # replaces experimental_rerun

elif st.session_state.input_mode == "camera":
    col_input, col_reset = st.columns([4,1])
    with col_input:
        image_input = st.camera_input("Take a photo of the fundus image")
    with col_reset:
        if st.button("Change Input"):
            st.session_state.input_mode = None
            st.stop()  # replaces experimental_rerun

# ==========================================================
#  ANALYSIS & RESULTS
# ==========================================================
if image_input is not None:
    image = Image.open(image_input).convert("RGB")

    col_image, col_retina_result, col_severity_result = st.columns(3)

    with col_image:
        st.markdown("<div class='image-display-card'>", unsafe_allow_html=True)
        st.subheader("1. Input Image")
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_retina_result:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("2. Retina Check")
        label, conf = predict_retina(image)
        st.markdown(f"**Classification:** {label.upper()}")
        st.metric("Confidence", f"{conf*100:.2f}%")
        if label.lower() == "retinal":
            st.success("✅ Image is suitable for DR screening.")
        else:
            st.error("❌ Non-retinal image. DR screening aborted.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_severity_result:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("3. DR Severity")
        if label.lower() == "retinal":
            sev_label, sev_conf = predict_severity(image)
            css_class = sev_label.replace(" ","").replace("-","")
            st.markdown(f"<p class='severity-level {css_class}'><strong>{sev_label.upper()}</strong></p>", unsafe_allow_html=True)
            st.metric("Prediction Confidence", f"{sev_conf*100:.2f}%")
            st.success("✅ Full diagnosis complete.")
        else:
            st.info("Awaiting valid retinal image.")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("⬆️ Select an input method above to start the diagnostic process.")
