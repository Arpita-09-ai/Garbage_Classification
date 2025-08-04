import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #32CD32);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #cccccc;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        padding: 3px;
        margin: 10px 0;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        height: 20px;
        border-radius: 7px;
        text-align: center;
        line-height: 20px;
        color: white;
        font-weight: bold;
    }
    
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
            font-size: 1.2rem;
        }
        .upload-section {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_PATH = "models/Waste_classifier_v2.h5"  # Relative path
IMG_SIZE = (224, 224)
CLASS_LABELS = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Enhanced class information
CLASS_INFO = {
    'cardboard': {'icon': '📦', 'tip': 'Flatten and keep dry before recycling', 'color': '#8B4513'},
    'glass': {'icon': '🍾', 'tip': 'Remove caps and rinse before recycling', 'color': '#00CED1'},
    'metal': {'icon': '🔩', 'tip': 'Clean cans and aluminum before recycling', 'color': '#C0C0C0'},
    'paper': {'icon': '📄', 'tip': 'Keep clean and dry for recycling', 'color': '#F5DEB3'},
    'plastic': {'icon': '🧴', 'tip': 'Check recycling number and clean before disposal', 'color': '#FF6347'},
    'trash': {'icon': '🗑️', 'tip': 'General waste - dispose in regular trash bin', 'color': '#696969'}
}

# Load model with error handling
@st.cache_resource
def load_trained_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Enhanced preprocessing
def preprocess_image(img: Image.Image) -> np.ndarray:
    try:
        img = img.convert("RGB").resize(IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Enhanced prediction with confidence visualization
def predict_image(img: Image.Image, model):
    processed = preprocess_image(img)
    if processed is None:
        return None, None
    
    predictions = model.predict(processed)
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions)
    return CLASS_LABELS[class_idx], confidence

# Main App Layout
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>♻️ Smart Waste Classification System</h1>
        <p>AI-powered waste sorting for a sustainable future</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar information
    with st.sidebar:
        st.header("📋 Classification Categories")
        for label, info in CLASS_INFO.items():
            st.markdown(f"""
            **{info['icon']} {label.title()}**  
            {info['tip']}
            """)
        
        st.header("📊 Model Information")
        st.info("**Accuracy:** 98%  \n**Model:** EfficientNetV2B2 + MobileNetV2")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="upload-section">
            <h3>📤 Upload Waste Image</h3>
            <p>Drag and drop or click to upload an image (JPG, PNG)</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of waste for classification"
        )

        # Camera option
        camera_image = st.camera_input("📷 Or take a photo")
        
        # Use camera image if uploaded_file is None
        if camera_image is not None and uploaded_file is None:
            uploaded_file = camera_image

    with col2:
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_column_width=True)
                
                # Load model
                model = load_trained_model()
                if model is None:
                    return
                
                # Auto-classify with loading animation
                with st.spinner('🔍 Analyzing image...'):
                    time.sleep(1)  # Simulate processing time
                    label, confidence = predict_image(img, model)
                
                if label and confidence:
                    # Results display
                    info = CLASS_INFO[label]
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h2 style="color: {info['color']};">{info['icon']} {label.title()}</h2>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence*100}%;">
                                {confidence*100:.1f}% Confidence
                            </div>
                        </div>
                        <p><strong>Recycling Tip:</strong> {info['tip']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional insights
                    if confidence > 0.9:
                        st.success("🎯 High confidence prediction!")
                    elif confidence > 0.7:
                        st.warning("⚠️ Moderate confidence - please verify")
                    else:
                        st.error("❌ Low confidence - try a clearer image")

            except Exception as e:
                st.error(f"Error processing image: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🌱 Help save the environment with proper waste classification</p>
        <p>Built with Streamlit • Powered by TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
