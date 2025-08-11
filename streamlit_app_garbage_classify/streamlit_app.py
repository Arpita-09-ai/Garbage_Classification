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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border: 3px dashed #667eea;
        text-align: center;
        margin: 1.5rem 0;
        color: #333333;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.25);
        border-color: #764ba2;
    }
    
    .upload-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: left 0.5s;
    }
    
    .upload-section:hover::before {
        left: 100%;
    }
    
    .upload-section h3 {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #4a5568;
    }
    
    .upload-section p {
        font-size: 1.1rem;
        color: #718096;
        margin: 0;
    }
    
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .result-card h2 {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .result-card p {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #4a5568;
        margin: 1rem 0;
    }
    
    .confidence-bar {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 15px;
        padding: 4px;
        margin: 15px 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #cbd5e1;
    }
    
    .confidence-fill {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        height: 25px;
        border-radius: 12px;
        text-align: center;
        line-height: 25px;
        color: white;
        font-weight: 700;
        font-size: 0.9rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .confidence-fill:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
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
    
    /* Enhanced page styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Beautiful sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader enhancement */
    .stFileUploader {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid #e2e8f0;
    }
    
    /* Camera input enhancement */
    .stCameraInput {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid #e2e8f0;
    }
    
    /* Success/Error message styling */
    .stSuccess, .stWarning, .stError {
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
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
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);">
        <h3 style="margin-bottom: 1rem; font-weight: 600;">🌱 Help save the environment with proper waste classification</h3>
        <p style="margin: 0.5rem 0; opacity: 0.9;">Built with Streamlit • Powered by TensorFlow</p>
        <div style="margin-top: 1rem; opacity: 0.7;">
            <span style="margin: 0 0.5rem;">♻️</span>
            <span style="margin: 0 0.5rem;">🌍</span>
            <span style="margin: 0 0.5rem;">💚</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
