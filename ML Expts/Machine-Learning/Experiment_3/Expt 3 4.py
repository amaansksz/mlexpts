import tkinter as tk
from tkinter import ttk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Function to perform multiple linear regression
def calculate_regression():
    try:
        # Get inputs from the user
        input_1 = float(entry_input1.get())
        input_2 = float(entry_input2.get())
        
        # Example data
        X = np.array([[1, 2], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
        y = np.array([3, 5, 9, 13, 17, 21])  # Target variable
        
        # Initialize and fit the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        new_input = np.array([[input_1, input_2]])
        prediction = model.predict(new_input)[0]
        
        # Calculate R² score
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Display results
        label_prediction.config(text=f"Prediction: {prediction:.2f}")
        label_r2.config(text=f"R² Score: {r2:.2f}")
        label_intercept.config(text=f"Intercept: {model.intercept_:.2f}")
        label_coefficients.config(text=f"Coefficients: {model.coef_}")
    except ValueError:
        label_error.config(text="Please enter valid numbers", fg="red")

# Create GUI window
root = tk.Tk()
root.title("Multiple Linear Regression(Amaan Shaikh S. 211P052)")

# Input fields for two numbers
label_input1 = ttk.Label(root, text="Enter first input (X1):")
label_input1.pack(pady=5)
entry_input1 = ttk.Entry(root)
entry_input1.pack(pady=5)

label_input2 = ttk.Label(root, text="Enter second input (X2):")
label_input2.pack(pady=5)
entry_input2 = ttk.Entry(root)
entry_input2.pack(pady=5)

# Button to trigger calculation
button_calculate = ttk.Button(root, text="Calculate", command=calculate_regression)
button_calculate.pack(pady=10)

# Labels to display results
label_prediction = ttk.Label(root, text="Prediction: ")
label_prediction.pack(pady=5)

label_r2 = ttk.Label(root, text="R² Score: ")
label_r2.pack(pady=5)

label_intercept = ttk.Label(root, text="Intercept: ")
label_intercept.pack(pady=5)

label_coefficients = ttk.Label(root, text="Coefficients: ")
label_coefficients.pack(pady=5)

label_error = ttk.Label(root, text="")
label_error.pack(pady=5)

# Start the GUI event loop
root.mainloop()
