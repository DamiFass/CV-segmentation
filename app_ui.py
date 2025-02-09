import torch
import torchvision
from PIL import Image
import numpy as np
import io
import os 
import gdown
import streamlit as st
from torchvision import transforms

# Model information
MODEL_URL = 'https://drive.google.com/uc?id=1wreniQXTbaYfSDE4mureC_DHGowh2Ds9'
MODEL_PATH = "trained_model.pt"

# Download the model if not present
if not os.path.exists(MODEL_PATH):
    st.write("Downloading the segmentation model... Please wait.")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.write("Model successfully downloaded! Ready to process images.")

# Load model
loaded_model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

def create_output_mask(prediction):
    """Generates a colored mask overlay for detected people."""
    predicted_dict = prediction[0]
    num_people = predicted_dict['labels'].shape[0]
    height, width = predicted_dict['masks'][0, 0].shape
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    colors = [
        [0, 0, 139], [0, 71, 171], [0, 0, 255], [100, 149, 237], 
        [0, 150, 255], [8, 143, 143], [115, 147, 179], [0, 255, 255],
        [125, 249, 255], [255, 165, 0]
    ]
    
    for i in range(num_people):
        rows, cols = np.where(predicted_dict['masks'][i, 0] >= 0.4)
        mask[rows, cols] = colors[i % len(colors)]
    
    return mask

# Image processing function
def predict(image, device):
    """Runs inference on the uploaded image."""
    image = image.convert("RGB")
    tensor_image = transforms.ToTensor()(image)
    with torch.no_grad():
        prediction = loaded_model([tensor_image.to(device)])
    return prediction

# Set device
DEVICE = torch.device('cpu')

# Streamlit UI
st.title("Pedestrian Segmentation App")
st.markdown(
    """Upload an image, and this app will automatically detect and segment people in it. 
    The model used is a fine-tuned Mask R-CNN, trained on the Penn-Fudan Pedestrian Database.
    """
)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

st.sidebar.header("About the Author")
st.sidebar.markdown("**Damiano Fassina**")
st.sidebar.markdown("[![LinkedIn](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/damianofassina/)")
st.sidebar.markdown("[![GitHub](https://img.icons8.com/material-outlined/48/000000/github.png)](https://github.com/DamiFass)")

if uploaded_file:
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown("**Click the button below to process the image.**")
    
    if st.button("Find People"): 
        prediction = predict(image, DEVICE)
        output_mask = create_output_mask(prediction)
        num_people = prediction[0]['labels'].shape[0]
        
        st.image(Image.fromarray(output_mask), caption="Segmented Output", use_column_width=True)
        st.markdown(f"### {num_people} {'person' if num_people == 1 else 'people'} detected in the image.")
else:
    st.info("Please upload an image to get started.")
