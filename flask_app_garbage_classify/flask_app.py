"""
flask_app.py
Smart Trash Classifier – Flask backend with enhanced UI, error-handling, and
mobile-friendly webcam support.


"""

import os
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import (
    Flask, render_template, request, redirect, flash,
    url_for, jsonify
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf


# ────────────────────────────
#  Configuration
# ────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.h5")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE_MB = 16                            # 16 MB upload limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "replace-with-your-own-secret"   # Needed for flash messages
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024


# ────────────────────────────
#  Model & Classes
# ────────────────────────────
try:
    model = load_model(MODEL_PATH, compile=False)
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] ✅  Model loaded from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] ❌  Model load failed: {e}")

CLASS_NAMES = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

TRASH_INFO = {
    "cardboard": "📦 Flatten boxes and keep them dry before recycling.",
    "glass":     "🍾 Remove lids & rinse. Glass is endlessly recyclable.",
    "metal":     "🔩 Rinse cans & foil; metals are highly recyclable.",
    "paper":     "📄 Keep clean and dry. Remove staples if possible.",
    "plastic":   "🧴 Check the resin code and rinse. Not all plastics recycle.",
    "trash":     "🗑️ General waste – dispose in the regular bin."
}


# ────────────────────────────
#  Utility Functions
# ────────────────────────────
def allowed_file(filename: str) -> bool:
    """Verify extension is jpg / jpeg / png."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def model_predict(img_path: str):
    """Return class label & confidence (0-100 %)."""
    if model is None:
        return "error", 0.0

    try:
        img = image.load_img(img_path, target_size=(224, 224))
        arr = image.img_to_array(img) / 255.0
        arr = tf.expand_dims(arr, 0)                     # (1, 224, 224, 3)
        preds = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(np.max(preds) * 100)               # 0-100 %
        return CLASS_NAMES[idx], conf
    except Exception as exc:
        print("Prediction error:", exc)
        return "error", 0.0


# ────────────────────────────
#  Routes
# ────────────────────────────
@app.route("/")
def root():
    return redirect(url_for("home"))


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # No file part?
        if "image" not in request.files:
            flash("No file part in form.", "error")
            return redirect(request.url)

        file = request.files["image"]

        # Empty filename?
        if file.filename == "":
            flash("Please choose an image file.", "error")
            return redirect(request.url)

        # Validate & save
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}"
            )
            file.save(save_path)

            # Predict
            label, conf = model_predict(save_path)
            if label == "error":
                flash("Prediction failed. Try again.", "error")
                return redirect(request.url)

            return render_template(
                "index.html",
                prediction=label,
                confidence=round(conf, 1),
                trash_info=TRASH_INFO.get(label, "No info available.")
            )

        flash("Unsupported file type. Upload PNG or JPG images.", "error")
        return redirect(request.url)

    # GET
    return render_template("index.html")


@app.route("/webcam")
def webcam():
    return render_template("webcam.html")


@app.route("/predict_webcam", methods=["POST"])
def predict_webcam():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image received"}), 400

    # Save under a constant name to overwrite each capture
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], "webcam_capture.jpg")
    file.save(save_path)

    label, conf = model_predict(save_path)
    if label == "error":
        return jsonify({"error": "Prediction failed"}), 500

    return jsonify(
        prediction=label,
        confidence=round(conf, 2),
        info=TRASH_INFO.get(label, "")
    )


# ────────────────────────────
#  Error Handlers
# ────────────────────────────
@app.errorhandler(413)
def file_too_large(e):
    flash(f"File exceeds {MAX_FILE_SIZE_MB} MB limit.", "error")
    return redirect(url_for("predict"))


@app.errorhandler(404)
def page_not_found(e):
    return render_template(
        "404.html",
        message="The page you are looking for does not exist."
    ), 404


# ────────────────────────────
#  Run
# ────────────────────────────
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,            # Turn off in production
        threaded=True
    )
