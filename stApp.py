# import torch
# import torchvision
from PIL import Image
import numpy as np
import io
# from torchvision.transforms import functional as F
# from torchvision import transforms
# from fastai.vision.widgets import *
# import ipywidgets as widgets
import streamlit as st

path = '../Downloads/trained_model'
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
    convert_tensor = transforms.ToTensor()
    bb = convert_tensor(my_image)
    with torch.no_grad():
        prediction = loaded_model([bb.to(device)])

    return prediction

device = torch.device('cpu')
    
st.write(""" ## Welcome:) Select the picture you want to segment! """)

uploaded_file = st.file_uploader('Your picture', label_visibility='hidden')

if uploaded_file:
    img = Image.open(io.BytesIO(uploaded_file.getvalue()))
    st.image(img, caption = '')
    st.write(""" #### Nice picture! Click on *Segment* to proceed :)""")

if st.button('Segment!'):
    
    if uploaded_file is None:
        st.write(""" Uploade a picture first :) """)
    else:

        prediction = predict(img, device)
        output_mask = create_output_mask(prediction)
        num_people = prediction[0]['labels'].shape[0]
        
        st.image(Image.fromarray(output_mask, 'RGB'))

        if num_people == 1:
            st.write(f' ### The system detected {num_people} person in the picture.')
        else:
            st.write(f' ### The system detected {num_people} people in the picture.')
