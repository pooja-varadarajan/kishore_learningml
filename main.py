from __future__ import division, print_function

import cv2
import os
import numpy as np

# Keras
import keras
from keras.models import load_model

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'bestmodel.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary

# print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    model = keras.models.load_model('bestmodel.h5')

    img = cv2.imread(img_path)[:, :, 0]
    img = cv2.resize(img, (28, 28))
   # cv2.imshow('image',img)
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    return str(np.argmax(prediction))



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)