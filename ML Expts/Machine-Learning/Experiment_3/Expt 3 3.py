import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Custom Linear Regression class
class MyLinearRegression:
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X_train, y_train):
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def predict(self, X_test):
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        return X_test @ np.r_[self.intercept_, self.coef_]

# Load the dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train inbuilt Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Train custom Linear Regression
my_model = MyLinearRegression()
my_model.fit(X_train, y_train)

# GUI setup
root = tk.Tk()
root.title("Linear Regression GUI(Amaan Shaikh S. 211P052)")

# Function to handle input and prediction
def run_model():
    try:
        feature_input = list(map(float, entry.get().split(',')))
        if len(feature_input) != X.shape[1]:
            raise ValueError("Input must have 10 values")

        feature_input = np.array([feature_input])

        # Predict with inbuilt model
        y_pred_builtin = model.predict(feature_input)

        # Predict with custom model
        y_pred_custom = my_model.predict(feature_input)

        messagebox.showinfo("Prediction Results",
                            f"Inbuilt Model Prediction: {y_pred_builtin[0]:.2f}\n"
                            f"Custom Model Prediction: {y_pred_custom[0]:.2f}")
    except ValueError as e:
        messagebox.showerror("Input Error", f"Invalid input: {e}")

# GUI elements
label = tk.Label(root, text="Enter 10 feature values separated by commas:")
label.pack(pady=10)

entry = tk.Entry(root, width=50)
entry.pack(pady=10)

button = tk.Button(root, text="Predict", command=run_model)
button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
