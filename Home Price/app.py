from flask import Flask, request, render_template
import pandas as pd

import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('D:/WebiSoftTech/MULTIPLELINEAR REGRESSION/home_price_model.pkl')

# Define a route for the homepage
@app.route('/')
def index():
    return render_template('D:/WebiSoftTech/MULTIPLELINEAR REGRESSION/template/index.html')

# Define a route to handle the prediction form
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data (replace 'feature1', 'feature2', etc. with actual feature names)
            feature1 = float(request.form['feature1'])
            feature2 = float(request.form['feature2'])
            feature3 = float(request.form['feature3'])

            # Prepare the input data for prediction
            data = {'feature1': [feature1], 'feature2': [feature2], 'feature3': [feature3]}
            df = pd.DataFrame(data)

            # Predict using the model
            prediction = model.predict(df)

            # Render the result
            return render_template('D:/WebiSoftTech/MULTIPLELINEAR REGRESSION/template/result.html', prediction=prediction[0])

        except Exception as e:
            return f"Error: {e}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
