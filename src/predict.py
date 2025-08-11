import os
import gradio as gr
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input # pyright: ignore[reportMissingImports]
from PIL import Image
import numpy as np
from src.config import SAVED_MODELS_PATH, MODEL_NAME, IMG_SIZE

# --- Load trained model ---
model_path = os.path.join(SAVED_MODELS_PATH, MODEL_NAME)
model = tf.keras.models.load_model(model_path) # pyright: ignore[reportAttributeAccessIssue]

# ✅ Keep class_names order identical to your training dataset folder order
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def classify_image(input_image):
    # 1. Resize to model input size
    img = input_image.resize(IMG_SIZE)

    # 2. Convert to numpy array (float32)
    img_array = np.array(img, dtype=np.float32)

    # 3. Add batch dimension → (1, H, W, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Preprocess for EfficientNetV2
    processed_array = preprocess_input(img_array)

    # 5. Predict
    predictions = model.predict(processed_array) # pyright: ignore[reportOptionalMemberAccess]

    # 6. Get highest-confidence class
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100  # % score

    # 7. Return as {label: confidence}
    return {predicted_class: confidence / 100}  # Gradio expects [0,1] for label confidence


# --- Gradio UI ---
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type='pil', label="Upload Trash Image"),
    outputs=gr.Label(num_top_classes=1, label="Prediction"),
    title="Trash Classifier ♻️",
    description="Upload an image of trash, and the model will predict its type with confidence.",
    examples=[
        ["examples/cardboard.jpg"],
        ["examples/plastic_bottle.jpg"],
    ]
)

if __name__ == "__main__":
    iface.launch()
