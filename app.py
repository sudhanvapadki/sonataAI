from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
import numpy as np
import librosa
from skimage.transform import resize
import io
import os

app = Flask(__name__)
CORS(app)

def create_model():
    inputs = Input(shape=(210, 210, 1))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)  # Changed from 32 to 64 filters
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create the model
model = create_model()

# Load the weights
try:
    model.load_weights('trained_model.keras')
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {str(e)}")
    raise  # This will stop the application if the model can't be loaded

# ... rest of your Flask app code ...

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")
    if 'file' not in request.files:
        print("No file part in the request")
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            print(f"Processing file: {file.filename}")
            file_content = io.BytesIO(file.read())
            X_test = load_and_preprocess_file(file_content)
            
            c_index = model_prediction(X_test)
            predicted_genre = classes[c_index]
            print(f"Predicted genre: {predicted_genre}")
            
            return jsonify({'genre': predicted_genre})
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
