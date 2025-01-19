import numpy as np
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import io
import cv2
from sklearn.preprocessing import LabelEncoder
import pickle

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model('model/arabic_handwritten.h5')

# Load the label encoder
with open('model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Arabic characters mapping
arabic_chars = {
    '0': 'ا',
    '1': 'ب',
    '2': 'ت',
    '3': 'ث',
    '4': 'ج',
    '5': 'ح',
    '6': 'خ',
    '7': 'د',
    '8': 'ذ',
    '9': 'ر',
    '10': 'ز',
    '11': 'س',
    '12': 'ش',
    '13': 'ص',
    '14': 'ض',
    '15': 'ط',
    '16': 'ظ',
    '17': 'ع',
    '18': 'غ',
    '19': 'ف',
    '20': 'ق',
    '21': 'ك',
    '22': 'ل',
    '23': 'م',
    '24': 'ن',
    '25': 'ه',
    '26': 'و',
    '27': 'ي'
}

def preprocess_image(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    # Resize
    img = cv2.resize(img, (32, 32))
    # Normalize
    img = img / 255.0
    # Flatten
    img = img.reshape(1, 32 * 32)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    
    # Preprocess the image
    processed_image = preprocess_image(image_bytes)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = str(np.argmax(prediction))
    confidence = float(np.max(prediction))
    
    # Convert numeric prediction to Arabic character
    arabic_char = arabic_chars.get(predicted_class, 'Unknown')
    
    return {
        "predicted_class": arabic_char,
        "confidence": confidence
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 