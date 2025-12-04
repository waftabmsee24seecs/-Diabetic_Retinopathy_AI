


import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import werkzeug

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.efficientnet import preprocess_input


# ==========================================================
#  SET MIXED PRECISION TO MATCH TRAINING
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


# ==========================================================
#  INITIALIZE FLASK APP
# ==========================================================
app = Flask(__name__)
CORS(app)

confidence = None

# Models loaded once
severity_model = None
retina_model = None


# ==========================================================
#  RETINA vs NON-RETINA MODEL (MODEL 1)
# ==========================================================
def load_retina_model():
    print("Loading retina vs non-retina model...")
    model = keras.models.load_model("retina_model_final.h5")
    print("Retina model loaded.")
    return model


def predicter(path, cnn_model):
    labels = ['non-retinal', 'retinal']
    IMG_SIZE = (300, 300)

    img = cv2.imread(str(path))
    if img is None:
        return "non-retinal"

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)

    img = np.expand_dims(img, axis=0)

    pred = cnn_model.predict(img)
    global confidence
    confidence = float(np.max(pred))

    return labels[int(np.argmax(pred))]


# ==========================================================
#  DIABETIC RETINOPATHY SEVERITY MODEL (MODEL 2)
# ==========================================================
def load_severity_model():
    print("Loading DR severity model...")

    # Load saved architecture
    with open("b4_model_architecture.json", "r") as f:
        json_arch = f.read()

    model = keras.models.model_from_json(
        json_arch,
        custom_objects={"QWK_Metric": QWK_Metric}
    )

    # Load weights
    model.load_weights("b4_final.weights.h5")
    print("Severity model loaded successfully.")

    return model


# ==========================================================
#  SEVERITY PREDICTION FUNCTION
# ==========================================================
def secondPredicter(path, model):

    labels = [
        'No DR',
        'Mild DR',
        'Moderate DR',
        'Severe DR',
        'Proliferative DR'
    ]

    IMG_SIZE = (300, 300)

    img = cv2.imread(str(path))
    if img is None:
        return "No DR"

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)

    # EfficientNetB4 preprocess
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    global confidence
    confidence = float(np.max(pred))

    return labels[int(np.argmax(pred))]


# ==========================================================
#  ROUTES
# ==========================================================
@app.route("/", methods=["GET"])
def home():
    return "Diabetic Retinopathy Prediction API is Running!"


# ----------------------------------------------------------
# ROUTE 1 : RETINA / NON-RETINA
# ----------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload():

    global retina_model
    if retina_model is None:
        retina_model = load_retina_model()

    imagefile = request.files["image"]
    filename = werkzeug.utils.secure_filename(imagefile.filename)

    if not os.path.exists("uploadedimages"):
        os.makedirs("uploadedimages")

    img_path = os.path.join("uploadedimages", filename)
    imagefile.save(img_path)

    label = predicter(img_path, retina_model)

    return jsonify({
        "confidence": confidence,
        "message": label
    })


# ----------------------------------------------------------
# ROUTE 2 : DR SEVERITY PREDICTION
# ----------------------------------------------------------
@app.route("/uploadS", methods=["POST"])
def uploadS():

    global severity_model
    if severity_model is None:
        severity_model = load_severity_model()

    imagefile = request.files["image"]
    filename = werkzeug.utils.secure_filename(imagefile.filename)

    if not os.path.exists("uploadedimages"):
        os.makedirs("uploadedimages")

    img_path = os.path.join("uploadedimages", filename)
    imagefile.save(img_path)

    label = secondPredicter(img_path, severity_model)

    return jsonify({
        "confidence": confidence,
        "message": label
    })


# ==========================================================
#  START THE SERVER
# ==========================================================
if __name__ == "__main__":

    if not os.path.exists("uploadedimages"):
        os.makedirs("uploadedimages")

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=True
    )