import subprocess
import sys
packages = [
        "opencv-python-headless",
        "streamlit",
        "numpy",
        "scikit-image",
        "pillow"
    ]
    
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    print("All packages installed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error installing packages: {e}")


import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Load and preprocess image
def load_image(image_file):
    """Loads an image from an uploaded file and converts it to grayscale."""
    img = Image.open(image_file).convert("RGB")  # Ensure image is in RGB format
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img, gray

# Resize image to match dimensions
def resize_images(img1, img2):
    """Resizes the second image to match the first image's size."""
    height, width = img1.shape[:2]
    img2_resized = cv2.resize(img2, (width, height))
    return img2_resized

# Compute Structural Similarity Index (SSIM)
def compare_images_ssim(image1, image2):
    """Computes SSIM (Structural Similarity Index) between two images."""
    ssim_value, _ = ssim(image1, image2, full=True)
    return ssim_value

# Detect progression (increase or decrease in affected area)
def detect_area_change(img1, img2):
    """Calculates the change in affected area based on pixel intensity difference."""
    img1_binary = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY)[1]
    img2_binary = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)[1]

    area1 = np.sum(img1_binary == 255)
    area2 = np.sum(img2_binary == 255)

    change = ((area2 - area1) / max(area1, 1)) * 100  # Avoid division by zero
    return round(change, 2)

# Display images side by side
def visualize_images(img1, img2):
    """Visualizes the uploaded images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1)
    axes[0].set_title("Before")
    axes[0].axis("off")

    axes[1].imshow(img2)
    axes[1].set_title("After")
    axes[1].axis("off")

    st.pyplot(fig)

# Custom CSS for UI enhancement
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .stTitle {
            text-align: center;
            font-size: 28px;
            color: #007bff;
        }
        .stButton>button {
            background-color: #28a745;
            color: white;
            font-size: 16px;
        }
        .stFileUploader>div>div {
            border: 2px dashed #007bff;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("🩺 Skin Condition Progression Tracker")
st.write("Upload two images to compare changes in skin condition.")

# Upload images
col1, col2 = st.columns(2)
with col1:
    image1 = st.file_uploader("Upload Initial Image", type=["jpg", "png", "jpeg"], key="img1")
with col2:
    image2 = st.file_uploader("Upload Recent Image", type=["jpg", "png", "jpeg"], key="img2")

if image1 and image2:
    # Load images from memory
    img1, gray1 = load_image(BytesIO(image1.getvalue()))
    img2, gray2 = load_image(BytesIO(image2.getvalue()))

    # Resize for comparison
    gray2_resized = resize_images(gray1, gray2)

    # Compute SSIM (similarity score)
    ssim_value = compare_images_ssim(gray1, gray2_resized)
    similarity_percentage = round(ssim_value * 100, 2)

    # Detect progression in affected area
    area_change = detect_area_change(gray1, gray2_resized)

    # Determine match status
    is_match = ssim_value > 0.5
    match_text = "✅ Condition Stable" if is_match else "⚠️ Condition Worsened"
    match_color = "#28a745" if is_match else "#D9534F"

    # Display results
    st.markdown(f"<h2 style='color: {match_color};'>SSIM Score: {similarity_percentage}%</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {match_color};'>{match_text}</h3>", unsafe_allow_html=True)
    st.write(f"### Estimated Change in Affected Area: {area_change}%")

    # Show images
    visualize_images(img1, img2)
