import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import joblib

# Load your dataset
data = {
    "area": [2600, 3000, 3200, 3600, 4000],
    "bedrooms": [3, 4, None, 3, 5],
    "age": [20, 15, 18, 30, 8],
    "price": [550000, 565000, 610000, 595000, 760000]
}
df = pd.DataFrame(data)

# Handle missing values
imputer = SimpleImputer(strategy='median')
df['bedrooms'] = imputer.fit_transform(df[['bedrooms']])

# Define features and target
X = df[['area', 'bedrooms', 'age']]
y = df['price']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
model_path = 'D:/WebiSoftTech/MULTIPLELINEAR REGRESSION/home_price_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
