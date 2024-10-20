import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(0)
# Generate random data
n_samples = 100
age = np.random.randint(18, 70, n_samples)
gender = np.random.choice(['Male', 'Female'], n_samples)
usage_minutes = np.random.randint(100, 1000, n_samples)
churn_status = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 80% not churned, 20% churned
# Create a DataFrame
df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'usage_minutes': usage_minutes,
    'churn_status': churn_status
})
df.to_csv('telecom_data_by_aditya.csv', index=False)
# Draw pair plot
sns.pairplot(df, hue='churn_status')
plt.suptitle('Pair Plot of Telecom Data Visualization By Amaan Shaikh S. 211P052', y=1.02)
plt.show()
