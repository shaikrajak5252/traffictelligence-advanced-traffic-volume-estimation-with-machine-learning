# Traffic-related-project-using-ai
# traffic_volume_prediction.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
data = pd.read_csv("traffic.csv")
# Display first 5 rows
print(data.head())
# Convert date column to datetime
data["date_time"] = pd.to_datetime(data["date_time"])

# Extract useful features
data["hour"] = data["date_time"].dt.hour
data["day"] = data["date_time"].dt.day
data["month"] = data["date_time"].dt.month
data["weekday"] = data["date_time"].dt.weekday

# Encode categorical columns
data = pd.get_dummies(data, columns=["weather_main"], drop_first=True)
# Features and target
X = data.drop(["traffic_volume", "date_time"], axis=1)
y = data["traffic_volume"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
# Save model
joblib.dump(model, "traffic_model.pkl")
print("Model saved successfully!")
