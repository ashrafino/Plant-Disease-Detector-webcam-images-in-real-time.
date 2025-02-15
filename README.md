# Plant Disease Detector webcam images in real-time. 🌿🔍
 This is a Plant Disease Detection web application built with Streamlit and TensorFlow. It utilizes a deep learning model to identify plant diseases from webcam images in real-time.
# Plant Disease Detector 🌿🔍

This is a **Plant Disease Detection** web application built with **Streamlit** and **TensorFlow**. It utilizes a deep learning model to identify plant diseases from webcam images in real-time.

## 🚀 Features
- **Real-time plant disease detection** using a pre-trained model.
- **Webcam integration** for live prediction.
- **Supports 38 plant disease classes**.
- **Uses MobileNetV2 preprocessing** for efficient performance.

## 🖥️ Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/YOUR_USERNAME/plant-disease-detector](https://github.com/ashrafino/Plant-Disease-Detector-webcam-images-in-real-time.git
   cd plant-disease-detector
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Download the trained model and place it in the project directory:
   ```sh
   mv best_model.h5 plant_disease_model.h5
   ```

4. Run the application:
   ```sh
   streamlit run main.py
   ```

## 📂 Files Structure
```
plant-disease-detector/
│── main.py               # Streamlit web app
│── requirements.txt      # Required Python packages
│── best_model.h5        # Pre-trained deep learning model
```

## 🔧 Dependencies
- Streamlit 1.32.2
- TensorFlow 2.16.1
- OpenCV 4.9.0.80
- NumPy 1.26.4

## 📝 Usage
1. Start the application.
2. Check the **Start Webcam** checkbox.
3. Point your camera at a plant leaf to get predictions.

## 📜 License
This project is open-source and available under the MIT License.

## 🙌 Credits
Developed by **Achraf El Bachra**.

