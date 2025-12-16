# =========================================================
# Diabetic Retinopathy AI Diagnostic Tool (CPU Optimized)
# =========================================================

import os
import io
import uuid
import datetime
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.efficientnet import preprocess_input

# CPU Profiling
import psutil
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False

# =========================================================
# GLOBAL CONFIG
# =========================================================

IMG_SIZE = (300, 300)  # input size for models
UPLOAD_DIR = "uploadedimages"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mixed precision for CPU (optional, may help TF 2.20)
mixed_precision.set_global_policy("mixed_float16")
print("✅ Mixed precision enabled (FP16)")

# =========================================================
# QWK METRIC
# =========================================================

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
            y_true, y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32
        )
        self.cm.assign_add(m)

    def result(self):
        cm = self.cm
        w = tf.zeros_like(cm)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                w = tf.tensor_scatter_nd_update(
                    w, [[i, j]],
                    [(i - j) ** 2 / (self.num_classes - 1) ** 2]
                )
        act = tf.reduce_sum(cm, 1)
        pred = tf.reduce_sum(cm, 0)
        expected = tf.tensordot(act, pred, axes=0) / tf.reduce_sum(cm)
        return 1.0 - tf.reduce_sum(w * cm) / tf.reduce_sum(w * expected)

    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))

# =========================================================
# IMAGE UTILITIES
# =========================================================

def bytes_to_cv2(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def save_image(image_bytes, filename=None):
    if not filename:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"camera_{ts}_{uuid.uuid4().hex[:6]}.png"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path

# =========================================================
# MODEL LOADING (CACHED)
# =========================================================

@st.cache_resource
def load_retina_model():
    start = time.time()
    model = keras.models.load_model("retina_model_final.h5", compile=False)
    end = time.time()
    return model, end - start

@st.cache_resource
def load_severity_model():
    start = time.time()
    with open("b4_model_architecture.json", "r") as f:
        model = keras.models.model_from_json(f.read(), custom_objects={"QWK_Metric": QWK_Metric})
    model.load_weights("b4_final.weights.h5")
    end = time.time()
    return model, end - start

RETINA_MODEL, RETINA_LOAD_TIME = load_retina_model()
SEVERITY_MODEL, SEVERITY_LOAD_TIME = load_severity_model()

# =========================================================
# CPU/GPU RESOURCE LOGGING
# =========================================================

def log_resource_usage():
    cpu_total = psutil.cpu_percent(interval=None)
    cpu_cores = psutil.cpu_percent(interval=None, percpu=True)
    mem = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    ram_process = process.memory_info().rss / 1024**2  # MB

    gpu_mem = gpu_util = 0
    if NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_mem = meminfo.used / 1024**2
            gpu_util = util.gpu
        except:
            pass

    return cpu_total, cpu_cores, mem.used / 1024**2, ram_process, gpu_mem, gpu_util

# =========================================================
# INFERENCE FUNCTIONS
# =========================================================

def predict_retina(image_bytes):
    labels = ["non-retinal", "retinal"]
    img = bytes_to_cv2(image_bytes)
    if img is None:
        return {"message":"non-retinal","confidence":0.0}
    img = cv2.resize(img, IMG_SIZE).astype(np.float32)
    img = np.expand_dims(img,0)
    pred = RETINA_MODEL(img, training=False).numpy()[0]
    idx = int(np.argmax(pred))
    return {"message": labels[idx], "confidence": float(pred[idx]*100)}

def predict_severity(image_bytes):
    labels = ["No DR","Mild DR","Moderate DR","Severe DR","Proliferative DR"]
    img = bytes_to_cv2(image_bytes)
    if img is None:
        return {"message":"No DR","confidence":0.0}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE).astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img,0)
    pred = SEVERITY_MODEL(img, training=False).numpy()[0]
    idx = int(np.argmax(pred))
    return {"message": labels[idx], "confidence": float(pred[idx]*100)}

# =========================================================
# PERFORMANCE PROFILING
# =========================================================

def profile_pipeline(image_bytes):
    logs = []

    # Retina Prediction
    t0 = time.time()
    retina_res = predict_retina(image_bytes)
    t1 = time.time()
    cpu_total, cpu_cores, ram_total, ram_process, gpu_mem, gpu_util = log_resource_usage()
    logs.append({
        "Stage": "Retina Prediction",
        "Time (s)": round(t1 - t0,3),
        "CPU Total (%)": cpu_total,
        "CPU Per Core (%)": ", ".join([f"Core{i+1}:{c}%" for i,c in enumerate(cpu_cores)]),
        "RAM Total (MB)": round(ram_total,1),
        "RAM Process (MB)": round(ram_process,1),
        "GPU Mem (MB)": round(gpu_mem,1),
        "GPU Util (%)": gpu_util
    })

    # Severity Prediction
    severity_res = None
    if retina_res['message']=="retinal":
        t0 = time.time()
        severity_res = predict_severity(image_bytes)
        t1 = time.time()
        cpu_total, cpu_cores, ram_total, ram_process, gpu_mem, gpu_util = log_resource_usage()
        logs.append({
            "Stage": "Severity Prediction",
            "Time (s)": round(t1 - t0,3),
            "CPU Total (%)": cpu_total,
            "CPU Per Core (%)": ", ".join([f"Core{i+1}:{c}%" for i,c in enumerate(cpu_cores)]),
            "RAM Total (MB)": round(ram_total,1),
            "RAM Process (MB)": round(ram_process,1),
            "GPU Mem (MB)": round(gpu_mem,1),
            "GPU Util (%)": gpu_util
        })

    # Total
    total_time = sum([l["Time (s)"] for l in logs])
    logs.append({
        "Stage": "Total Pipeline",
        "Time (s)": round(total_time,3),
        "CPU Total (%)": cpu_total,
        "CPU Per Core (%)": ", ".join([f"Core{i+1}:{c}%" for i,c in enumerate(cpu_cores)]),
        "RAM Total (MB)": round(ram_total,1),
        "RAM Process (MB)": round(ram_process,1),
        "GPU Mem (MB)": round(gpu_mem,1),
        "GPU Util (%)": gpu_util
    })

    return retina_res, severity_res, logs

# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="Diabetic Retinopathy AI Diagnostic", layout="wide")

if "input_mode" not in st.session_state:
    st.session_state.input_mode = None

st.markdown("""<style>/* Optional CSS */</style>""", unsafe_allow_html=True)

st.title("👁️ Diabetic Retinopathy AI Diagnostic Tool")
st.markdown("Two-stage DR diagnosis with elegant result display and detailed performance metrics.")

with st.sidebar:
    st.header("About This Demo")
    st.info("Two-stage DR diagnosis using AI on CPU.")
    st.markdown(f"**Retina Model Load Time:** {RETINA_LOAD_TIME:.2f}s  \n**Severity Model Load Time:** {SEVERITY_LOAD_TIME:.2f}s")

# ==================== Input Selection ====================

st.subheader("Select Image Input Method")
image_input = None
if st.session_state.input_mode is None:
    c1,c2 = st.columns(2)
    if c1.button("⬆️ Upload Image"): st.session_state.input_mode="file"; st.rerun()
    if c2.button("📸 Capture Image"): st.session_state.input_mode="camera"; st.rerun()
elif st.session_state.input_mode=="file":
    image_input=st.file_uploader("Choose image", ["jpg","jpeg","png"])
    if st.button("Change Input"): st.session_state.input_mode=None; st.rerun()
elif st.session_state.input_mode=="camera":
    image_input=st.camera_input("Capture image")
    if st.button("Change Input"): st.session_state.input_mode=None; st.rerun()

# ==================== Display Columns ====================

col_img, col_r, col_s = st.columns([1,1,1], gap="medium")  # centered layout

if image_input:
    image_bytes=image_input.read()
    path = save_image(image_bytes,getattr(image_input,"name",None))
    pil_img = Image.open(io.BytesIO(image_bytes)).resize((400,400))  # smaller image

    with col_img:
        st.subheader("1. Input Image")
        st.image(pil_img, width=400)
        st.caption(f"Saved to: {path}")

    with col_r:
        st.subheader("2. Retina Check")
        retina_res = predict_retina(image_bytes)
        st.markdown(f"""
        <div style='padding:10px; border:2px solid #4CAF50; border-radius:10px;'>
            <h4 style='color:#4CAF50;'>Retina Check: {retina_res['message'].title()}</h4>
            <div style='background:#ddd; border-radius:5px;'>
                <div style='width:{retina_res['confidence']}%; background:#4CAF50; color:white; text-align:center; border-radius:5px;'>
                    {retina_res['confidence']:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_s:
        st.subheader("3. DR Severity")
        if retina_res['message']=="retinal":
            severity_res = predict_severity(image_bytes)
            st.markdown(f"""
            <div style='padding:10px; border:2px solid #f44336; border-radius:10px;'>
                <h4 style='color:#f44336;'>Severity: {severity_res['message']}</h4>
                <div style='background:#ddd; border-radius:5px;'>
                    <div style='width:{severity_res['confidence']}%; background:#f44336; color:white; text-align:center; border-radius:5px;'>
                        {severity_res['confidence']:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Non-retinal image")
            severity_res = None

    # ==================== Performance & Logs ====================
    st.subheader("🖥️ Performance & Resource Usage")
    retina_res, severity_res, logs = profile_pipeline(image_bytes)

    # Show model load times
    logs_with_models = [
        {"Stage":"Retina Model Load","Time (s)": round(RETINA_LOAD_TIME,3), "CPU Total (%)":"-", "CPU Per Core (%)":"-", "RAM Total (MB)":"-", "RAM Process (MB)":"-", "GPU Mem (MB)":"-", "GPU Util (%)":"-"},
        {"Stage":"Severity Model Load","Time (s)": round(SEVERITY_LOAD_TIME,3), "CPU Total (%)":"-", "CPU Per Core (%)":"-", "RAM Total (MB)":"-", "RAM Process (MB)":"-", "GPU Mem (MB)":"-", "GPU Util (%)":"-"},
    ] + logs

    for log in logs_with_models:
        st.markdown(f"""
        <div style='padding:5px 10px; border-bottom:1px solid #ccc;'>
        <b>{log['Stage']}</b>: {log['Time (s)']} s
        <br>CPU Total: {log['CPU Total (%)']} | CPU Per Core: {log['CPU Per Core (%)']}
        <br>RAM Total: {log['RAM Total (MB)']} MB | RAM Process: {log['RAM Process (MB)']} MB
        <br>GPU Mem: {log['GPU Mem (MB)']} MB | GPU Util: {log['GPU Util (%)']} %
        </div>
        """, unsafe_allow_html=True)

else:
    with col_img:
        st.subheader("1. Input Image"); st.info("Upload image to start")
    with col_r:
        st.subheader("2. Retina Check"); st.info("Awaiting image")
    with col_s:
        st.subheader("3. DR Severity"); st.info("Awaiting Stage 1")
    st.subheader("🖥️ Performance & Resource Usage"); st.info("Awaiting profiling")

st.markdown("---")
st.caption("⚠️ For demonstration only. Not for medical use.")
