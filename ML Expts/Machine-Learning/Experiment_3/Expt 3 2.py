import tkinter as tk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('C:/Users/amaan/Desktop/ML Expts/Machine-Learning/Experiment_3/add.csv')

# Features (x) and target (y)
x = data[['x', 'y']]
y = data['sum']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Model training
model = LinearRegression()
model.fit(x_train, y_train)

# Function to handle inputs and prediction
def display_inputs():
    # Retrieve inputs from the entry widgets
    input1 = float(entry1.get())
    input2 = float(entry2.get())
    
    # Make predictions using the trained model
    y_pred = model.predict([[input1, input2]])[0]
    
    # Display the result in the label
    result_label.config(text=f"Input 1: {input1}\nInput 2: {input2}\nSum(Prediction): {y_pred:.2f}")

# GUI setup using Tkinter
root = tk.Tk()
root.title("ML Prediction of Sums(Amaan Shaikh S. 211P052)")

# Input field 1
label1 = tk.Label(root, text="Enter Input 1:")
label1.pack(pady=5)
entry1 = tk.Entry(root)
entry1.pack(pady=5)

# Input field 2
label2 = tk.Label(root, text="Enter Input 2:")
label2.pack(pady=5)
entry2 = tk.Entry(root)
entry2.pack(pady=5)

# Button to display the inputs and prediction
display_button = tk.Button(root, text="Add", command=display_inputs)
display_button.pack(pady=10)

# Label to display the prediction
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
