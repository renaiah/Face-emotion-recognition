import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import os
import cv2

# Load the model from the saved file
model = models.load_model('E:/ML projects/Face Emotion detection/model.keras') 

emotions = [['angry'],
            ['disgust'],
            ['fear'],
            ['happy'],
            ['neutral'],
            ['sad'],
            ['surprise']]

st.header('Human Emotion Recognition')
img_path = st.text_input('Enter the image path')

if img_path:
    image = cv2.imread(img_path)[:,:,0]
    image = cv2.resize(image, (48,48))
    image = np.invert(np.array([image]))

    output = np.argmax(model.predict(image))

    outcome = emotions[output]

    stn = 'Emotion in the image is : ' + str(outcome[0])

    st.markdown(stn)

    img_name = os.path.basename(img_path)
    st.image(img_path, width=300)  



