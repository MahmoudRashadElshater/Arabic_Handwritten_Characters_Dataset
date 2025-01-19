import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import kagglehub

def load_data(data_path):
    # Load images and labels from CSV files
    train_images = pd.read_csv(os.path.join(data_path, 'csvTrainImages 13440x1024.csv'), header=None).values
    train_labels = pd.read_csv(os.path.join(data_path, 'csvTrainLabel 13440x1.csv'), header=None).values
    test_images = pd.read_csv(os.path.join(data_path, 'csvTestImages 3360x1024.csv'), header=None).values
    test_labels = pd.read_csv(os.path.join(data_path, 'csvTestLabel 3360x1.csv'), header=None).values

    # Combine training and testing data
    images = np.concatenate([train_images, test_images])
    labels = np.concatenate([train_labels, test_labels])

    # Reshape images to 32x32
    images = images.reshape(-1, 32, 32)

    # Normalize pixel values
    images = images / 255.0

    return images, labels.flatten()

def create_model():
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(1024,)),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(28, activation='softmax')  # 28 Arabic characters
    ])
    return model

def main():
    # Create necessary directories
    os.makedirs('model', exist_ok=True)
    
    # Download and load data
    path = kagglehub.dataset_download("mloey1/ahcd1")
    X, y = load_data(path)
    
    # Flatten images
    X = X.reshape(-1, 32 * 32)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = tf.keras.utils.to_categorical(y_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
    
    # Create and compile model
    model = create_model()
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Train model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
    
    # Save model and label encoder
    model.save('model/arabic_handwritten.h5')
    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print("Model and label encoder saved successfully!")

if __name__ == "__main__":
    main() 