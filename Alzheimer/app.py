from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask import request, jsonify
from PIL import Image


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = "Alzheimer_MRI.h5"

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):

    import keras
    import numpy as np
    from tensorflow.keras.preprocessing import image
 
    image = image.load_img(img_path)
    image = image.resize((128, 128))
    image = np.array(image)
    image = image.reshape(1, 128, 128, 3)

    prediction = model.predict(image)

    predicted_class = np.argmax(prediction)

    LLL = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
    predic_clss = LLL[predicted_class]
    # Return the predictions
    return predic_clss


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        result = str(preds)

        # Return the result as JSON response
        return result

if __name__ == '__main__':
    app.run(debug=True)
