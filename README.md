# air-digit-recognition-cnn
Handwritten digit recognition using CNN on MNIST with real-time air-writing support

✋ **Air Digit Recognition using CNN (MNIST + Hand Tracking)**

Convolutional Neural Network for handwritten digit recognition with real-time “Air Writing” support using a webcam

📌 **Project Overview**

This project implements a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits.
In addition, the project extends beyond static classification by enabling real-time “air digit” recognition using webcam input and hand tracking.

Users can draw digits in the air using their fingertip. The motion is captured, processed, converted into a 28×28 grayscale image, and passed to a trained CNN to predict the digit.

🎯 **Objective**

To design a deep learning system that:

Learns digit patterns from image data

Works in real-time with webcam input

Demonstrates how computer vision and deep learning can work together

Simulates real-world gesture-based ML systems

🛠 **Tech Stack**

Python
TensorFlow / Keras
NumPy
Matplotlib & Seaborn
OpenCV
MediaPipe

📊 **Dataset**

MNIST Handwritten Digits Dataset

60,000 training images

10,000 testing images

Image size: 28 × 28 grayscale

Loaded directly using keras.datasets

🧠 **Model Architecture**

The CNN model follows this design:

Convolutional Layer (ReLU)

Max Pooling

Convolutional Layer

Max Pooling

Flatten Layer

Fully Connected Dense Layer

Output Layer with Softmax (10 classes)

Loss Function: sparse_categorical_crossentropy
Optimizer: Adam
Metric: Accuracy

The model achieved ~99% accuracy on the test dataset.

✋ **Real-Time Air Digit Recognition**

This project includes:

✅ Hand landmark detection using MediaPipe
✅ Drawing canvas with OpenCV
✅ Digit preprocessing pipeline:

Cropping
Resizing
Normalization
✅ Real-time prediction via CNN

📈 **Results & Evaluation**

Training and validation accuracy monitored over epochs

Confusion matrix visualization

Classification report (Precision, Recall, F1-Score)

Test accuracy close to 99%

📁 **Project Structure**

Since this project is notebook-based:

air-digit-recognition/

│── air_digit_prediction.ipynb

│── README.md

▶ **How to Run**
1. Install dependencies:
pip install tensorflow opencv-python mediapipe numpy matplotlib seaborn

2. Open the notebook:
jupyter notebook air_digit_cnn.ipynb

3. Execute all cells

Make sure your webcam is connected and permissions are allowed.

🚀 **Future Improvements**

Convert into a Streamlit web app

Add confidence scores

Train on custom handwritten images

Improve digit segmentation accuracy

Add UI controls

👤 **Author**

Srikanth Gunti
📧 Email: srikanthgunti11@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/srikanth-gunti-

⭐ Feedback

If you find this project useful, feel free to ⭐ star the repository!
