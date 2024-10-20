# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
# Replace '<your-dataset-filename>.csv' with the actual dataset filename
data = pd.read_csv('C:/Users/amaan/Desktop/ML Expts/Machine-Learning/Experiment_10/Breast_cancer_data.csv')

# Step 3: Inspect the dataset (optional)
print(data.head())
print(data.info())
print(data.describe())

# Step 4: Prepare the data
# Assuming 'target' is the column for benign/malignant (0 = benign, 1 = malignant)
X = data.drop('diagnosis', axis=1)  # Features
y = data['diagnosis']  # Target variable (benign or malignant)

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create and train the AdaBoost classifier
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = ada_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Step 8: Visualize Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap(Amaan Shaikh S. 211P052)')
plt.show()

# Step 9: Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix(Amaan Shaikh S. 211P052)')
plt.show()

# Step 10: Create a function for prediction
def predict_breast_cancer(inputs):
    input_df = pd.DataFrame([inputs], columns=X.columns)  # Create a DataFrame with the same column names as X
    prediction = ada_model.predict(input_df)
    return prediction[0]  # Return the prediction result

# Step 11: Take user inputs for prediction (based on features in the dataset)
# Example: Replace these with actual values from your dataset features
print('Breast Cancer report by Amaan Shaikh S. 211P052')
mean_radius = float(input("Enter mean radius: "))
mean_texture = float(input("Enter mean texture: "))
mean_perimeter = float(input("Enter mean perimeter: "))
mean_area = float(input("Enter mean area: "))
mean_smoothness = float(input("Enter mean smoothness: "))

# Add as many features as needed from your dataset

# Prepare inputs for prediction
user_inputs = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]

# Step 12: Make the prediction
prediction = predict_breast_cancer(user_inputs)

# Step 13: Output the prediction
if prediction == 1:
    print("Malignant tumor detected.")
else:
    print("Benign tumor detected.")