# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load your dataset
data = pd.read_csv('C:/Users/amaan/Desktop/ML Expts/Machine-Learning/Experiment_10/WineQT.csv')

# Step 3: Check for any missing values
print(data.isnull().sum())  # Ensure no missing values; if any, handle them

# Step 4: Prepare your data
# 'quality' is the target variable in this case
X = data.drop('quality', axis=1)  # Features
y = data['quality']  # Target variable

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create and train the ElasticNet model
elastic_net_model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
elastic_net_model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = elastic_net_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('ElasticNet report by Amaan Shaikh S. 211P052:')
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Step 8: Visualize Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap(Amaan Shaikh S. 211P052)')
plt.show()

# Step 9: Visualize Residuals (Actual vs. Predicted)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Quality(Amaan Shaikh S. 211P052)')
plt.show()

# Step 10: Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True, color='blue')
plt.title('Residual Plot(Amaan Shaikh S.)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
