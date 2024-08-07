import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('D:\\ezitech\\model1.h5')

# Function to predict gender from image
def male_or_female(img_path, model):
    img = load_img(img_path, target_size=(70, 80))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    new_array_with_batch_dimension = np.expand_dims(img_array, axis=0)
    output = model.predict(new_array_with_batch_dimension)
    output = np.round(output.flatten())[0]
    val = "Male" if output == 1 else 'Female'
    return val

# Streamlit app
st.title("Gender Prediction Model")

# Upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Save the uploaded image to a temporary location
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Make a prediction when the button is clicked
    if st.button("Predict"):
        prediction = male_or_female("temp_image.jpg", model)
        st.write(f"The predicted gender is: {prediction}")

