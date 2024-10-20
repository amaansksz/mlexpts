import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load Iris dataset
iris = load_iris()

# Create DataFrame from the Iris dataset
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

# Split the DataFrame into different classes
df0 = df[:50]      # Setosa
df1 = df[50:100]   # Versicolor
df2 = df[100:]     # Virginica

# Set up the plotting environment
# %matplotlib inline # Uncomment if using Jupyter or Google Colab

# Plot Sepal Length vs Sepal Width for different classes
plt.figure(figsize=(10, 5))  # Set figure size for better visibility
plt.title("Amaan Shaikh 211P052\nPlot Sepal Length vs Sepal Width")
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='green', marker='+', label="Setosa")
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='blue', marker='x', label="Versicolor")
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color='red', marker='o', label="Virginica")  # Add Virginica for completeness
plt.legend()
plt.show()

# Plot Petal Length vs Petal Width for different classes
plt.figure(figsize=(10, 5))  # Set figure size for better visibility
plt.title("Amaan Shaikh 211P052\nPlot Petal Length vs Petal Width")
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='green', marker='+', label="Setosa")
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue', marker='x', label="Versicolor")
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color='red', marker='o', label="Virginica")  # Add Virginica for completeness
plt.legend()
plt.show()

# Split the data into features (X) and target (y)
X = df.drop(columns=['target', 'flower_name'])
y = df.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train SVM model (default parameters)
model = SVC()
model.fit(X_train, y_train)

# Make a prediction with the default SVM model
print("Prediction for [4.8, 3.0, 1.5, 0.3]: ", model.predict([[4.8, 3.0, 1.5, 0.3]]))

# Train and compare models with different parameters
# SVM model with C=1
model_C_1 = SVC(C=1)
model_C_1.fit(X_train, y_train)

# SVM model with C=10
model_C_10 = SVC(C=10)
model_C_10.fit(X_train, y_train)

# SVM model with gamma=10
model_gamma_10 = SVC(gamma=10)
model_gamma_10.fit(X_train, y_train)

# SVM model with linear kernel
model_linear = SVC(kernel='linear')
model_linear.fit(X_train, y_train)
