# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Reading the CSV file and loading into 'df' object
df = pd.read_csv('C:/Users/amaan/Desktop/ML Expts/Machine-Learning/Experiment_5/adult_dataset.csv')

# Data Information
df.info()
df.head()

# Check for missing values represented as '?'
df.isin(['?']).sum(axis=0)

# Replace '?' with NaN and drop rows with NaN values
df = df.replace('?', np.nan)
df = df.dropna()

# Label encoding for categorical variables
le = LabelEncoder()
df = df.apply(le.fit_transform)

# Splitting the data into features and target variable
x = df.drop('income', axis=1)
y = df['income']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Creating and training the Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Making predictions on the test set
y_pred = clf.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plotting the Decision Tree
plt.title("Amaan Shaikh S. 211P052")
plt.figure(figsize=(10, 8))
plot_tree(decision_tree=clf, feature_names=x.columns, class_names=['<=50k', '>50k'], filled=True)
plt.show()
