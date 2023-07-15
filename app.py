import base64
from flask import Flask, request, jsonify, render_template
import tensorflow as tf #for the model and CNN
import numpy as np
from PIL import Image, ImageChops, ImageEnhance #for ELA functions
import math
import requests #for testing
import werkzeug
import cv2 #for CBIR functions
import os
#==============================
image_size = (128, 128)

app = Flask(__name__)

# Load your trained CNN model
model = tf.keras.models.load_model('trueshot_model_3c7.h5')

# Define a list of class labels/names
class_names = ['Tampered', 'Authentic', 'AI-Generated']

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image foundsfds'})

    image = request.files['image']
    filename = werkzeug.utils.secure_filename(image.filename)
    image.save("./uploadedimages/" + image.filename)
    return jsonify({'image_name': image.filename})


# API endpoint for classification
@app.route('/classify', methods=['POST'])
def classify():
    # Check if the request contains an image
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image_path = request.files['image']
    
    # Load the image and preprocess it
    image = prepare_image(image_path)
    image = image.reshape(-1, 128, 128, 3)
    
    # Make predictions
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions, axis = 1)[0]
    predicted_class_name = class_names[predicted_class_index]
    # model.summary()

    print(predicted_class_name)
    confidence = np.amax(predictions) * 100
    print(confidence)
    final_predictions = (str(math.trunc(confidence)) + "% " + predicted_class_name)
    return jsonify({'class_label': predicted_class_name})

#Function for preparing an image
def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

#Function for Converting to ELA in Model A
def convert_to_ela_image(path, quality):
    global ela_filename2, ela_filename3
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.jpg'

    image = Image.open(path)
    image = image.convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)

    temp_image = Image.open(temp_filename)
    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    ela_image.save(ela_filename, 'JPEG', quality = quality)
    return ela_image


#===========================CONTENT BASED IMAGE RETRIEVAL 
# Function to extract image features
def extract_features(image):
    # Perform feature extraction (e.g., convert to grayscale, extract histograms, etc.)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    features = hist.flatten()
    return features

# Function to calculate similarity between two feature vectors
def calculate_cosine_similarity(query_features, image_features):
    # Calculate the dot product of the two vectors
    dot_product = np.dot(query_features, image_features)

    # Calculate the norms of the two vectors
    query_norm = np.linalg.norm(query_features)
    image_norm = np.linalg.norm(image_features)

    # Calculate the cosine similarity
    cosine_similarity = dot_product / (query_norm * image_norm)
    return cosine_similarity

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


@app.route('/cbir', methods=['POST'])
def cbir():
    if 'query_image' not in request.json or 'dataset_images' not in request.json:
        return 'Missing query image or dataset images in the request', 400
    # Get the query image and dataset images from the request payload


    query_image_string = request.json['query_image']
    print("Query Image String: ", query_image_string)
    query_image = download_image(query_image_string)

    dataset_image_strings = request.json['dataset_images']
    print("Dataset Image Strings: ", dataset_image_strings)


    query_image = download_image(query_image_string)
    # Convert the query image string to an OpenCV image
    # query_image = base64.b64decode(query_image_string)
    # query_image = cv2.imdecode(np.frombuffer(query_image_bytes, np.uint8), cv2.IMREAD_COLOR)

    if query_image is None:
        return 'Failed to download the query image', 400

    # Extract features from the query image
    query_features = extract_features(query_image)

    # Process dataset images
    similarities = []
    for image_string in dataset_image_strings:
        # Convert the image string to an OpenCV image
        image = download_image(image_string)
        # image = cv2.imdecode(np.frombuffer(image_string, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f'Failed to download the image: {image_string}')
            continue
        image_features = extract_features(image)
        similarity = calculate_cosine_similarity(query_features, image_features)
        similarities.append(similarity)

    top_k = 4
    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    # Prepare the response data to send back to Flutter
    results = [dataset_image_strings[index] for index in top_k_indices]
    print("Similar Images: ", results)
    return jsonify({'Similar Images': results})

if __name__ == '__main__':
    app.run(debug=True)


