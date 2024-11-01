import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create a sample dataset
np.random.seed(42)
n_samples = 1000

data = {
    'temperature': np.random.uniform(10, 35, n_samples),
    'rainfall': np.random.uniform(500, 1500, n_samples),
    'humidity': np.random.uniform(30, 90, n_samples),
    'soil_quality': np.random.uniform(0, 10, n_samples),
    'fertilizer': np.random.uniform(50, 200, n_samples),
    'crop_yield': np.zeros(n_samples)
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Generate crop yield based on other features (simplified model)
df['crop_yield'] = (
    2 * df['temperature'] +
    0.01 * df['rainfall'] +
    0.5 * df['humidity'] +
    100 * df['soil_quality'] +
    0.5 * df['fertilizer'] +
    np.random.normal(0, 50, n_samples)  # Add some noise
)

# Save the dataset
df.to_csv('crop_yield_dataset.csv', index=False)

# Now, let's proceed with the data analysis and prediction
# Load the dataset
df = pd.read_csv('crop_yield_dataset.csv')

# Display basic information about the dataset
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Plot correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
plt.imshow(correlation_matrix, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix of Crop Yield Features')
plt.tight_layout()
plt.show()

# Prepare the data for modeling
X = df.drop('crop_yield', axis=1)
y = df['crop_yield']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Plot feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.title('Feature Importance in Crop Yield Prediction')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Crop Yield')
plt.ylabel('Predicted Crop Yield')
plt.title('Actual vs Predicted Crop Yield')
plt.tight_layout()
plt.show()

# Predict crop yield for a new sample
new_sample = np.array([[25, 1000, 60, 7, 150]])  # Example values
new_sample_scaled = scaler.transform(new_sample)
predicted_yield = rf_model.predict(new_sample_scaled)
print(f"\nPredicted crop yield for the new sample: {predicted_yield[0]:.2f}")
