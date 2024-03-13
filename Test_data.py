import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv('Data_collection.csv')

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the test dataset to a CSV file
test_df.to_csv('test_dataset.csv', index=False)

# Load the trained model
model = joblib.load('trained_model.pkl')

# Read the test dataset
test_df = pd.read_csv('test_dataset.csv')

# Split the test dataset into features (X_test) and target variable (y_test)
X_test = test_df.drop('Sale_Price', axis=1)
y_test = test_df['Sale_Price']

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error: {}'.format(mse))

# Visualize the predictions
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Price')
plt.show()
