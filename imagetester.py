import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    # Load the image from the given path
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert the image to an array
    img_array = image.img_to_array(img)

    # Expand dimensions to match the model's input shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Rescale the image (if necessary, depending on your model's preprocessing)
    img_array /= 255.0  # Assuming you rescaled images during training

    return img_array

# Function to make a prediction
def predict_image(model, img_path):
    # Preprocess the image
    img_array = load_and_preprocess_image(img_path)

    # Make a prediction
    prediction = model.predict(img_array)

    # Threshold the prediction to classify as fake or real
    if prediction >= 0.5:
        result = "Fake"
    else:
        result = "Real"

    return result

# Load your trained model
model = tf.keras.models.load_model('F:\\third year mini project codes\deepfake_detectioncnn2.keras')

# Predict whether an image is fake or real
image_path = 'F:\\third year mini project codes\\testing imgs\\ypqclxhnhgal9nyohyst.jpg'  # Replace with the path of the image you want to test
result = predict_image(model, image_path)

# Print the result
print(f'The image is classified as: {result}')
