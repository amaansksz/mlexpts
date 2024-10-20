import pandas as pd
from sklearn.model_selection import train_test_split


data = {
    'mileage': [150000, 30000, 45000, 60000, 75000, 90000, 105000, 120000],
    'price': [15000, 2000, 10000, 8000, 6000, 4000, 3000, 12000]
}
# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)


# Split the dataset into training and test sets
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
# Display the training and test sets
print("Training set:\n",train_set)
print("\nTest set:\n",test_set)
print('Amaan Shaikh S. 211P052')
