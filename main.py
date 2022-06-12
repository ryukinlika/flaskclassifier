from operator import methodcaller
import os
from app import app, bootstrap
from predict_classes import mix_classes, local_classes
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import pandas as pd
import joblib
import tensorflow as tf
import keras
import numpy as np
from keras_cv_attention_models import coatnet


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload_form():
	return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()

@app.route('/inference', methods=['POST'])
def inference(): 
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    filename = request.form.get('file_name')
    if filename == '':
        flash('File name empty')
        return redirect("/")

    print(filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # loada classifier
    modelType = request.form.get('model')
    if modelType == "coatnet":
        model = keras.models.load_model("model/mix/3_coatnet_sgd_normal")
    elif modelType == "efficientnet":
        model = keras.models.load_model("model/mix/1_efficientnet_adam_brightness_blur.h5", compile=False)
    else: #resnet
        model = keras.models.load_model("model/mix/4_resnet_adam_normal.h5", compile=False)

    # prepare image
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img))
    if modelType != "coatnet": # model Efficientnet and ResNet
        img = (img - 0.5) / 0.5  # value to -1:1
    print("============IMAGE SHAPE=============")
    print(img.shape)
    resized_img = tf.image.resize(img, [224, 224])
    resized_img = np.expand_dims(resized_img, axis=0)
    print(resized_img.shape)

    # get prediction
    prediction = model.predict(resized_img)
    
    predict_label = mix_classes[np.argmax(prediction)]

    return render_template('inference.html', filename=filename, prediction=predict_label, model=modelType)
