import os
from app import app, bootstrap
from predict_classes import  local_classes
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import keras


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
PROJECT_NAME = "nema-online-classifier"
PROJECT_LOCATION = "asia-northeast1"
PROJECT_ENDPOINT_ID = "7422108107767021568"
PROJECT_API_ENDPOINT = "asia-northeast1-aiplatform.googleapis.com"

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/inference', methods=['POST'])
def inference_image():
	if 'file' not in request.files:
		flash('File part is missing')
		return render_template('inference.html')
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return render_template('inference.html')
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		file.save(path)
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
		
		model = keras.models.load_model("model/effv2b0.h5")
		
		img = tf.io.read_file(path)
		img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
		img = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img))
		img = (img - 0.5) / 0.5  

		resized_img = tf.image.resize(img, [224, 224])
		resized_img = np.expand_dims(resized_img, axis=0)
		
        #get prediction
		res = model.predict(resized_img)

		percentage = round(np.amax(res)*100.0, 2)
		inference = local_classes[np.argmax(res)]
        
		return render_template('inference.html', filename=filename, inference=inference, percentage=percentage)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return render_template('inference.html')

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()

