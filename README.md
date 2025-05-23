# Handwritten_digitreco
# 🖥️ Handwritten Digit Recognition using KNN (with Streamlit)

This project is a simple digit recognition web app that uses the **K-Nearest Neighbors (KNN)** algorithm to classify handwritten digits from images. The app is built using **Python**, **NumPy**, **Matplotlib**, and **Streamlit**. It uses the **`sklearn.datasets.load_digits()`** dataset (8×8 images of digits).

---

## 🔧 Features

- Upload your own digit image (PNG/JPG/JPEG)
- Automatically preprocesses the image to match the dataset format
- Predicts the digit using a trained KNN model
- Visualizes the uploaded and processed images
- Shows prediction accuracy on test samples
- Displays reference images for digits 1 to 9

---

## 🧠 Technologies Used

| Library       | Purpose                            |
|---------------|------------------------------------|
| **NumPy**     | Array and image data manipulation  |
| **Matplotlib**| Image visualization and plotting   |
| **Pillow**    | Image preprocessing and resizing   |
| **Streamlit** | Building the interactive web app   |
| **scikit-learn** | KNN model, dataset, evaluation  |

---

## 📂 Dataset

This project uses the **`load_digits()`** dataset from `sklearn`, which includes:
- 8×8 grayscale images of handwritten digits (0–9)
- 64 features per image (flattened 8×8 pixels)
- 1,797 total samples

---

## ⚙️ How It Works

1. The dataset is split into training and testing sets.
2. You choose the number of neighbors (`K`) via the sidebar.
3. A KNN model is trained using the selected K.
4. You can upload a digit image for prediction:
   - It is converted to grayscale
   - Resized to 8×8 pixels
   - Inverted if needed (white background check)
   - Normalized to pixel values between 0–16
   - Flattened and passed to the model
5. The app predicts the digit and displays the result.

---

## 🖼️ Screenshots

### Sample Image Upload and Prediction
![Sample Upload](screenshots/upload.png)

### Sample Predictions from Test Dataset
![Test Predictions](screenshots/test_predictions.png)

---

## 🏁 Getting Started

### ✅ Prerequisites
Install required libraries:
```bash
pip install streamlit numpy matplotlib scikit-learn pillow
