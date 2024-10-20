import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
data={
    'customer':[1,2,3,4,5],
    'age':[25,45,np.nan,35,50],
    'gender':['Male','Female','Female',np.nan,'Male'],
    'annual_income':[50000,60000,45000,80000,120000],
    'purchase_amount':[200,150,300,400,np.nan],
    'purchase_date':['2023-01-01','2023-02-15','2023-02-15','2023-02-15','2023-02-15']
}
df=pd.DataFrame(data)
print("Original DataFrame (Amaan Shaikh S. 211P052)\n",df)
# Handling missing values
df['age'].fillna(df['age'].mean(),inplace=True)
df['gender'].fillna(df['gender'].mode()[0],inplace=True)
df['purchase_amount'].fillna(df['purchase_amount'].median(),inplace=True)
print("\nDataFrame after handling missing values:\n",df)
# Encoding categorical variables
df['gender']=df['gender'].map({'Male':0,'Female':1})
print("\nDataFrame after encoding categorical variables:\n",df)
# Normalizing numerical variables
scaler=StandardScaler()
df[['age','annual_income','purchase_amount']]=scaler.fit_transform(df[['age','annual_income','purchase_amount']])
print("\nDataFrame after normalizing numerical variables:\n",df)
# Creating a new features - total purchase amount per customer
df['total_purchase_amount']=df['purchase_amount'].cumsum()
print("\nDataFrame after creating a new feature:\n",df)
