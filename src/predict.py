import os
import gradio as gr
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image
import numpy as np
from src.config import SAVED_MODELS_PATH, MODEL_NAME, IMG_SIZE

# --- 1. Load the saved model and class names ---
model_load_path = os.path.join(SAVED_MODELS_PATH, MODEL_NAME)
model = tf.keras.models.load_model(model_load_path)
# Make sure this list is in the same order as your training data folders
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'] 


def classify_image(input_image):
    
    # 1. Resize image to the size the model expects (124x124)
    img = input_image.resize(IMG_SIZE)
    
    # 2. Convert image to a numpy array of numbers
    img_array = np.array(img, dtype=np.float32)
    
    # 3. Add a "batch" dimension. The model expects (1, 124, 124, 3) not (124, 124, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Preprocess the image 
    processed_array = preprocess_input(img_array)

    # 5. Make a prediction
    predictions = model.predict(processed_array)
    
    # 6. Find the class with the highest score
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]
    
    # 7. Return the result in a nice format
    return {predicted_class: float(confidence)}

# --- 2. Create the Gradio Interface ---

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type='pil', label="Upload Trash Image"),
    outputs=gr.Label(num_top_classes=1, label="Prediction"),
    title="Trash Classifier ♻️",
    description="Upload an image of trash, and the model will predict its type.",
    examples=[
        ["examples/cardboard.jpg"],
        ["examples/plastic_bottle.jpg"],
    ] # Optional: create an 'examples' folder with some images
)

iface.launch() # Use share=True to get a public link