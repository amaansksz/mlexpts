import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import KMeans

# Ignore warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('C:/Users/amaan/Desktop/ML Expts/Machine-Learning/Experiment_8/Mall_Customers.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Display information about the DataFrame
df.info()

# Select the features for clustering (3rd and 4th columns)
x = df.iloc[:, [3, 4]]  # Assuming the 3rd column is 'Annual Income' and the 4th is 'Spending Score'
print(x.head())


# Initialize a list to store the WCSS (Within-Cluster Sum of Squares) values
wcss = []

# Compute WCSS for different numbers of clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plot the elbow method
plt.figure(figsize=(10, 5))  # Set figure size for better visibility
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method(Amaan Shaikh S. 211P052)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))  # Set x-ticks for clarity
plt.grid()
plt.show()
