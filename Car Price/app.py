from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score

# Load dataset
carprices_data = pd.read_csv("D:/WebiSoftTech/MULTIPLELINEAR REGRESSION/Car Price/carprices.csv")

# One-hot encoding for Car Model
encoder = OneHotEncoder()
encoded_car_model = encoder.fit_transform(carprices_data[['Car Model']]).toarray()
encoded_columns = encoder.get_feature_names_out(['Car Model'])

# Combine encoded car models with the dataset
encoded_df = pd.DataFrame(encoded_car_model, columns=encoded_columns)
processed_data = pd.concat([carprices_data.drop(columns=['Car Model']), encoded_df], axis=1)

# Features and target
X = processed_data.drop(columns=['Sell Price($)'])
y = processed_data['Sell Price($)']

# Scale numerical features
scaler = StandardScaler()
X[['Mileage', 'Age(yrs)']] = scaler.fit_transform(X[['Mileage', 'Age(yrs)']])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = r2_score(y_test, model.predict(X_test))

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Car Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    car_model = data['Car Model']
    mileage = data['Mileage']
    age = data['Age(yrs)']

    # Encode car model
    car_model_encoded = encoder.transform([[car_model]]).toarray()
    other_features = np.array([[mileage, age]])
    other_features_scaled = scaler.transform(other_features)

    # Combine features
    input_features = np.hstack((other_features_scaled, car_model_encoded))

    # Predict
    predicted_price = model.predict(input_features)[0]
    return jsonify({"Predicted Price": predicted_price})

@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    return jsonify({"Model Accuracy (R2 Score)": accuracy})

if __name__ == '__main__':
    app.run(debug=True)
