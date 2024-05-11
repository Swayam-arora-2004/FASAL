import matplotlib
matplotlib.use('Agg')
from flask import Flask,request,render_template,url_for,redirect,jsonify
import tensorflow as tf
import numpy as np
import pandas
import sklearn
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import joblib
# creating flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# importing model
new_model = tf.keras.models.load_model('trained_plant_disease_model.h5')
model = joblib.load('crop_recommendation_model.pkl')
scaler = joblib.load('scaler.pkl')

# sc = pickle.load(open('standscaler.pkl','rb'))
# ms = pickle.load(open('minmaxscaler.pkl','rb'))

label_encoder = {0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut', 5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean', 15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas', 19: 'pomegranate', 20: 'rice', 21: 'watermelon'}
def get_crop_recommendation(N, P, K, temperature, humidity, ph, rainfall, model):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_data_scaled = scaler.transform(input_data)
    print("Input data scaled:", input_data_scaled)
    predicted_label = model.predict(input_data_scaled)
    print("Predicted label:", predicted_label)
    crop_label = label_encoder[predicted_label[0]]
    print("Crop label:", crop_label)
    return crop_label


#####
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about',methods=['GET'])
def about():
    return render_template("about.html")

@app.route('/contact',methods=['GET'])
def contact():
    return render_template("contact.html")

@app.route('/login',methods=['GET'])
def login():
    return render_template("login-signup.html")

@app.route('/recommend',methods=['GET','POST'])
def recommend():
    return render_template("crop-recommendation.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        N = request.form['Nitrogen']
        P = request.form['Phosporus']
        K = request.form['Potassium']
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])
        recommended_crop = get_crop_recommendation(N, P, K, temp, humidity, ph, rainfall, model)
        result = "{} is the best crop to be cultivated right there".format(recommended_crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('crop-recommendation.html',result = result)

@app.route('/damage',methods=['GET','POST'])
def damage():
    return render_template("crop-damage.html")

@app.route('/analysis', methods=['POST'])
def analysis():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Preprocess the image
            img = tf.keras.preprocessing.image.load_img(filename, target_size=(128, 128))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            # Make prediction
            prediction = new_model.predict(img_array)
            predicted_class = np.argmax(prediction)

            # Load class names
            with open('class_names.json', 'r') as f:
                class_names = json.load(f)

            # Visualize result
            plt.imshow(img)
            plt.title(class_names[str(predicted_class)])
            plt.axis('off')
            plt.savefig('static/result.png')  # Save the visualization
            plt.close()

            return render_template('crop-damage.html', prediction=class_names[str(predicted_class)])
# python main
if __name__ == "__main__":
    app.run(debug=True)
