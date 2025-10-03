import streamlit as st
import joblib
import pickle
from PIL import Image
import numpy as np

with open("svm_image_classifier_model.pkl", "rb") as f:
    model = joblib.load(f)

st.title("Fruit Classifier")
st.write(" ")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

class_dict = {0: "แอปเปิ้ล", 1: "ส้ม"} 

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Predict'): 
        
        image_rgb = image.convert('RGB') 
    
        image_resized = image_rgb.resize((100, 100))
        
        image_array = np.array(image_resized)
        
        image_array_flat = image_array.flatten().reshape(1, -1)
        
        prediction = model.predict(image_array_flat)[0]
        
        prediction_name = class_dict[prediction]
    
        st.write(f"**ผลการทำนาย: {prediction_name}**")