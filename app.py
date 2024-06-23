import pandas as pd
from flask import Flask, render_template, request
import joblib

# Load the trained model
model = joblib.load("stroke_model/stroke.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        DHTTemp = float(request.form['DHTTemp'])
        DS18bTemp = float(request.form['DS18bTemp'])
        Humidity = float(request.form['Humidity'])
        BloodPre1 = float(request.form['BloodPre1'])
        BloodPre2 = float(request.form['BloodPre2'])
        BMP = float(request.form['BMP'])
        OxygenLevel = float(request.form['OxygenLevel'])
        Age = int(request.form['Age'])
        Gender = int(request.form['Gender'])
        Smoking = int(request.form['Smoking'])
        HeartDisease = int(request.form['HeartDisease'])
        Stroke = int(request.form['Stroke'])
        
        # Create a new DataFrame for prediction
        new_data = pd.DataFrame({
            'DHTTemp': [DHTTemp],
            'DS18bTemp': [DS18bTemp],
            'Humidity': [Humidity],
            'BloodPre1': [BloodPre1],
            'BloodPre2': [BloodPre2],
            'BMP': [BMP],
            'Oxygen Level (%)': [OxygenLevel],
            'Age': [Age],
            'Gender': [Gender],
            'Smoking': [Smoking],
            'Heart Disease History': [HeartDisease],
            'Stroke History': [Stroke]
        })
        
        # Use the model to make predictions
        predicted_value = model.predict(new_data)
        
        return render_template('result.html', prediction=predicted_value)

if __name__ == '__main__':
    app.run(debug=True)
