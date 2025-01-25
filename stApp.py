import torch
import torchvision
from PIL import Image
import numpy as np
import io
import os 
import requests
# from torchvision.transforms import functional as F
from torchvision import transforms
# from fastai.vision.widgets import *
# import ipywidgets as widgets
import streamlit as st


def download_model():
    model_url = "https://drive.google.com/file/d/1wreniQXTbaYfSDE4mureC_DHGowh2Ds9/view?usp=drive_link"
    model_path = "trained_model.pt"

    if not os.path.exists(model_path):
        print("Downloading model...")
        response = requests.get(model_url, stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded.")
    return model_path

# Download the model from cloud and load
path = download_model()
loaded_model = torch.load(path, map_location=torch.device('cpu'))


def create_output_mask(prediction):
    predicted_dict = prediction[0]
    num_people = predicted_dict['labels'].shape[0]

    height, width, channel = predicted_dict['masks'][0,0].shape[0], predicted_dict['masks'][0,0].shape[1], 3
    red, green, blue = 0, 0, 0
    arr = np.full((height, width, channel), [red, green, blue], dtype=('uint8'))

    shades = [ [0, 0, 139], [0, 71, 171], [0, 0, 255], [100, 149, 237], [0, 150, 255], [8, 143, 143], [115, 147, 179], 
              [0, 255, 255], [0, 255, 255], [125, 249, 255] ]

    for i in range(num_people):
        rows, cols = np.where( predicted_dict['masks'][i,0] >= 0.4 )
        arr[rows,cols,0] = shades[i][0]
        arr[rows,cols,1] = shades[i][1]
        arr[rows,cols,2] = shades[i][2]

    return arr

# Function to process the image to make inference on:
def predict(my_image, device):
    my_image = my_image.convert("RGB") 
    convert_tensor = transforms.ToTensor()
    bb = convert_tensor(my_image)
    with torch.no_grad():
        prediction = loaded_model([bb.to(device)])

    return prediction

device = torch.device('cpu')
    
st.write(""" ## Hey there:) Select the picture you want to find people on! """)

uploaded_file = st.file_uploader('Your picture', label_visibility='hidden')

st.sidebar.markdown(" #### Author: Damiano Fassina. ")
st.sidebar.markdown(" #### Find me on: ")
st.sidebar.markdown("[![Title](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/damianofassina/)")
st.sidebar.markdown("[![Title](https://img.icons8.com/material-outlined/48/000000/github.png)](https://github.com/DamiFass)")
                                
if uploaded_file:
    img = Image.open(io.BytesIO(uploaded_file.getvalue()))
    st.image(img, caption = '')
    st.write(""" #### Nice picture! Click on *Find people* to proceed :)""")

if st.button('Find people!'):
    
    if uploaded_file is None:
        st.write(""" Uploade a picture first :) """)
    else:

        prediction = predict(img, device)
        output_mask = create_output_mask(prediction)
        num_people = prediction[0]['labels'].shape[0]
        
        st.image(Image.fromarray(output_mask, 'RGB'))

        if num_people == 1:
            st.write(f' ### {num_people} person has been found in the picture.')
        else:
            st.write(f' ### {num_people} people have been found in the picture.')
