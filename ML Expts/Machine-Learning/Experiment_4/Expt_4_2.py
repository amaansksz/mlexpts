import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
file_path = 'C:/Users/amaan/Desktop/ML Expts/Machine-Learning/Experiment_4/titanic.csv'  # Updated file path
data = pd.read_csv(file_path)

# Display the first few rows and info of the dataset to understand its structure
print(data.head())
print(data.info())

# Visualizations
# Count plot of survivors
sns.countplot(x='Survived', data=data)
plt.title('Count of Survivors')
plt.show()

# Count plot of survivors by sex
sns.countplot(x='Survived', hue='Sex', data=data)
plt.title('Count of Survivors by Sex')
plt.show()

# Count plot of survivors by passenger class
sns.countplot(x='Survived', hue='Pclass', data=data)
plt.title('Count of Survivors by Passenger Class')
plt.show()

# Age distribution
data['Age'].plot.hist(bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.show()

# Fare distribution
data['Fare'].plot.hist(bins=20, figsize=(10, 5))
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.show()

# Siblings/Spouses aboard
sns.countplot(x='SibSp', data=data)
plt.title('Number of Siblings/Spouses Aboard')
plt.show()

# Heatmap to visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), yticklabels=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Data preprocessing
# Drop 'Cabin' as it has many missing values
data.drop('Cabin', axis=1, inplace=True)

# Drop rows with missing values
data.dropna(inplace=True)

# One-hot encoding for categorical columns: 'Sex', 'Embarked', 'Pclass'
male = pd.get_dummies(data['Sex'], drop_first=True)
embark = pd.get_dummies(data['Embarked'], drop_first=True)
pc1 = pd.get_dummies(data['Pclass'], drop_first=True)

# Update dataset with encoded columns and drop unnecessary columns
data = pd.concat([data, male, embark, pc1], axis=1)
data.drop(['Sex', 'Embarked', 'PassengerId', 'Pclass', 'Name', 'Ticket'], axis=1, inplace=True)

# Split the data into features (X) and target (y)
X = data.drop('Survived', axis=1)
y = data['Survived']

# Ensure all feature columns are numeric and column names are strings
X.columns = X.columns.astype(str)  # Ensure column names are strings
X = X.apply(pd.to_numeric, errors='coerce')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Logistic Regression
lr = LogisticRegression()

# Train the model
lr.fit(X_train, y_train)

# Make predictions
predictions = lr.predict(X_test)

# Evaluation
classification_rep = classification_report(y_test, predictions)
confusion_mat = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)

# Output the evaluation results
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", confusion_mat)
print("Accuracy: {:.2f}%".format(accuracy * 100))
