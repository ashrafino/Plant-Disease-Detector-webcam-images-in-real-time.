import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "plant_disease_model.h5"  # Use the exact filename from GitHub
CLASS_NAMES = [
    'Apple Scab', 'Apple Black Rot', 'Cedar Apple Rust', 'Apple Healthy',
    'Blueberry Healthy', 'Cherry Healthy', 'Cherry Powdery Mildew',
    'Corn Gray Leaf Spot', 'Corn Common Rust', 'Corn Healthy',
    'Grape Black Rot', 'Grape Esca', 'Grape Healthy', 'Grape Leaf Blight',
    'Orange Haunglongbing', 'Peach Bacterial Spot', 'Peach Healthy',
    'Pepper Bacterial Spot', 'Pepper Healthy', 'Potato Early Blight',
    'Potato Healthy', 'Potato Late Blight', 'Raspberry Healthy',
    'Soybean Healthy', 'Squash Powdery Mildew', 'Strawberry Healthy',
    'Strawberry Leaf Scorch', 'Tomato Bacterial Spot', 'Tomato Early Blight',
    'Tomato Healthy', 'Tomato Late Blight', 'Tomato Leaf Mold',
    'Tomato Septoria Leaf Spot', 'Tomato Spider Mites', 'Tomato Target Spot',
    'Tomato Mosaic Virus', 'Tomato Yellow Leaf Curl Virus'
]

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        # Force input shape for Streamlit compatibility
        model.layers[0]._batch_input_shape = (None, 224, 224, 3)
        model = tf.keras.models.model_from_config(model.get_config())
        model.set_weights(model.get_weights())
        return model
    except Exception as e:
        st.error(f"""Model loading failed! Ensure:
                 1. You've downloaded 'plant_disease_model.h5' from GitHub
                 2. File is in correct directory
                 3. Dependencies match (TensorFlow 2.7+)
                 Error: {str(e)}""")
        st.stop()

def preprocess_image(image):
    # Exact preprocessing from SAURABHSINGHDHAMI's implementation
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)  # MobileNetV2 specific preprocessing
    return np.expand_dims(image, axis=0)

def main():
    st.title("Plant Disease Detector 🌿🔍")
    model = load_model()
    
    # Webcam setup
    cap = cv2.VideoCapture(0)
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    DIAGNOSTIC = st.empty()

    while run:
        ret, frame = cap.read()
        if not ret: break
        
        # Preprocess and predict
        processed = preprocess_image(frame)
        pred = model.predict(processed)
        class_id = np.argmax(pred)
        confidence = np.max(pred)
        
        # Handle potential mismatches
        try:
            label = f"{CLASS_NAMES[class_id]} ({confidence:.2%})"
        except IndexError:
            label = f"Unknown (Class {class_id}, {confidence:.2%})"
        
        # Display
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(display_frame, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        FRAME_WINDOW.image(display_frame)
        
        # Show diagnostics
        DIAGNOSTIC.write(f"**Model Input:** 224x224 RGB | **Processing Time:** {pred[1]:.2f}s")

    cap.release()

if __name__ == "__main__":
    main()