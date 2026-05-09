import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

features = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','YearBuilt','YrSold','TotRmsAbvGrd']

X_train = pd.read_csv('data/X_train.csv')[features]
y_train = np.load('data/y_train.npy')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model = Ridge()
model.fit(X_scaled, y_train)

pickle.dump(model, open('backend/models/best_model.pkl', 'wb'))
pickle.dump(scaler, open('backend/models/scaler.pkl', 'wb'))
print('Models saved successfully!')