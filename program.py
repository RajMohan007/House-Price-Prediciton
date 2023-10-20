# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# Load your dataset (replace 'data.csv' with your dataset file)
data = pd.read_csv('synthetic_dataset.csv')

# Assuming your dataset contains features (X) and target variable (y)
X = data.drop('Price', axis=1)  # Features
y = data['Price']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

model.fit(X_train, y_train)



# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Now, you can use the trained model to make predictions for new data
# For example:
new_data = np.array([[1000,2,1,2000,3]])  # Replace with actual feature values
predicted_price = model.predict(new_data)
print(f"Predicted Price: {predicted_price[0]}")
