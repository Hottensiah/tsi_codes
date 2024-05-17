#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[3]:


import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
eth_usd_max = pd.read_csv('eth_usd_max.csv')

# Drop rows with missing values
eth_usd_max = eth_usd_max.dropna()

# Define the feature matrix (X) and the target vector (y)
X = eth_usd_max[['market_cap', 'total_volume']]
y = eth_usd_max['price']

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train[['market_cap', 'total_volume']] = scaler.fit_transform(X_train[['market_cap', 'total_volume']])
X_test[['market_cap', 'total_volume']] = scaler.transform(X_test[['market_cap', 'total_volume']])

# Fit the model using statsmodels
model = sm.OLS(y_train, X_train).fit()

# Print the summary of the regression
print(model.summary())


# In[ ]:




