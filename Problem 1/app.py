# Import necessary libraries
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

# Load the dataset
file_path = 'D:/WebiSoftTech/MULTIPLELINEAR REGRESSION/Problem 1/ca11-03homes.xls'

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load the dataset
data = pd.read_excel(file_path)
print("Dataset columns:", data.columns)

# Define features and target using actual column names
features = ['SqFt', 'BedRooms', 'Baths']
target = 'Price'

# Check if columns exist in the dataset
for col in features + [target]:
    if col not in data.columns:
        raise KeyError(f"Column '{col}' not found in the dataset.")

# Extract features and target
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Create the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict house prices based on input features.
    Expected input (JSON):
    {
      "houses": [
        {"sqft": 780, "bedrooms": 3, "bathrooms": 1},
        {"sqft": 1500, "bedrooms": 3, "bathrooms": 2}
      ]
    }
    """
    input_data = request.get_json()
    predictions = []

    # Iterate through each house and predict price
    for house in input_data['houses']:
        sqft = house.get('sqft', 0)
        bedrooms = house.get('bedrooms', 0)
        bathrooms = house.get('bathrooms', 0)
        prediction = model.predict([[sqft, bedrooms, bathrooms]])[0]
        predictions.append({
            'sqft': sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'predicted_price': prediction
        })

    return jsonify(predictions)

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
