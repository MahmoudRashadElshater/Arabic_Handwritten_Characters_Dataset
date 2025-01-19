import streamlit as st
import requests
from PIL import Image
import io

st.title("Arabic Handwritten Character Recognition")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Create a button to trigger prediction
    if st.button('Predict'):
        # Prepare the file for the API request
        files = {'file': ('image.jpg', uploaded_file.getvalue(), 'image/jpeg')}
        
        # Make prediction request to FastAPI backend
        try:
            response = requests.post('http://localhost:8000/predict', files=files)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Character: {result['predicted_class']}")
                st.info(f"Confidence: {result['confidence']:.2%}")
            else:
                st.error("Error making prediction. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the backend server. Please make sure it's running.") 