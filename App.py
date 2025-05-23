 import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sidebar settings
st.sidebar.header("🔧 Settings")
k = st.sidebar.slider("Select K (Neighbors)", 1, 10, 3)
num_samples = st.sidebar.slider("Number of Sample Digits", 5, 20, 10)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
knn.fit(X_train, y_train)

# Determine resample filter based on Pillow version
try:
    resample_filter = Image.Resampling.LANCZOS
except AttributeError:
    resample_filter = Image.ANTIALIAS

# App title
st.title("🖥 Handwritten Digit Recognition with KNN")

# Upload image section
st.header("📤 Upload a Digit Image (preferably black background, white digit)")

uploaded_file = st.file_uploader(
    "Upload image (PNG/JPG/JPEG)", 
    type=["png", "jpg", "jpeg"]
)

def preprocess_image(image):
    # Convert to grayscale
    image = ImageOps.grayscale(image)
    # Resize to 8x8 with proper resample filter
    image = image.resize((8, 8), resample_filter)
    # Convert to numpy array
    arr = np.array(image).astype(np.float32)
    # Invert colors if background is white (mean pixel high)
    if np.mean(arr) > 128:
        arr = 255 - arr
    # Normalize pixel values to 0-16 (like sklearn digits)
    arr = (arr / 255.0) * 16
    arr = np.clip(arr, 0, 16)
    # Flatten to 1D vector for prediction
    return arr.flatten().reshape(1, -1)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=150)

    processed_image = preprocess_image(image)
    
    # Show processed image
    st.write("🔍 Processed Image (8x8 grayscale):")
    fig, ax = plt.subplots()
    ax.imshow(processed_image.reshape(8, 8), cmap='gray')
    ax.axis('off')
    st.pyplot(fig)
    
    # Predict digit
    prediction = knn.predict(processed_image)
    st.success(f"🎯 Predicted Digit: {prediction[0]}")

# Show sample test predictions
st.header("📊 Sample Predictions from Test Dataset")
fig, axes = plt.subplots(2, num_samples // 2, figsize=(12, 6))
axes = axes.ravel()
sample_indices = np.random.choice(len(X_test), num_samples, replace=False)

for i, idx in enumerate(sample_indices):
    axes[i].imshow(X_test[idx].reshape(8, 8), cmap='gray')
    axes[i].set_title(f"True: {y_test[idx]} | Pred: {knn.predict([X_test[idx]])[0]}")
    axes[i].axis('off')

st.pyplot(fig)

# Show model accuracy in sidebar
accuracy = accuracy_score(y_test, knn.predict(X_test))
st.sidebar.markdown(f"✅ Model Accuracy: {accuracy:.4f}")

# Show digits 1 to 9 from dataset for reference
st.header("🔢 Digits 1 to 9 from Dataset")

fig2, axes2 = plt.subplots(1, 9, figsize=(15, 3))
for digit in range(1, 10):
    idx = np.where(y == digit)[0][0]
    axes2[digit-1].imshow(X[idx].reshape(8, 8), cmap='gray')
    axes2[digit-1].set_title(str(digit))
    axes2[digit-1].axis('off')

st.pyplot(fig2)
