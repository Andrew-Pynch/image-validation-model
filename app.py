##############################################
# Project put together by Andrew Pynch
# Email: andrewpynchbusiness@gmail.com
# Web: https://andrew-pynch.github.io/
##############################################

import numpy as np
from flask import Flask, request, render_template
import pickle

from resources.utils import *
from fastai.vision.widgets import *
from fastai.imports import *

import os

cwd = os.getcwd()
path = Path()
Path().ls(file_exts=".pkl")

application = Flask(__name__)

model = load_learner(path / "model/export.pkl")

# Defining the home page for the web service
@application.route("/")
def home():
    return render_template("index.html")


# Writing api for inference using the loaded model
@application.route("/predict", methods=["POST"])

# Predict method that uses the trained model to predict the kind of bear in the picture we uploaded
def predict():

    # labels = ['grizzly','black','teddy']

    file = request.files["file"]

    # Store the uploaded images in a temporary folder
    if file:
        filename = file.filename
        file.save(os.path.join("resources/tmp", filename))

    to_predict = "resources/tmp/" + filename

    # Getting the prediction from the model
    prediction = model.predict(to_predict)

    # Render the result in the html template
    return render_template(
        "index.html", prediction_text="Your Prediction :  {} ".format(prediction[0])
    )


if __name__ == "__main__":
    # run the application
    application.run(host="0.0.0.0")
