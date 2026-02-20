# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: R Venkatramani
RegisterNumber: 25010118    //   212225240182
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

df = pd.read_csv("encoded_car_data.csv")
print(df.head())

X = df[['horsepower','enginesize', 'citympg', 'highwaympg']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr=Pipeline([('scaler',StandardScaler()),
            ('model', LinearRegression())])
lr.fit(X_train, y_train)
y_pred_linear=lr.predict(X_test)

poly_model = Pipeline([('poly',PolynomialFeatures(degree=2)),
                      ('scaler',StandardScaler()),
                      ('model',LinearRegression())])
poly_model.fit(X_train,y_train)
y_pred_poly=poly_model.predict(X_test)


print("Name: R Venkatramani ")
print("Reg No: 212225240182 ")
print("Linear Regression:")

print(f"{'MSE'}: {mean_squared_error(y_test,y_pred_linear)}")
r2score=r2_score(y_test,y_pred_linear)
print("R2 Score=",r2score)

print("\nPolynomial Regression:")
print(f"{'MSE'}: {mean_squared_error(y_test,y_pred_poly)}")

print(f"{'R-squared'}: {r2_score(y_test,y_pred_poly)}")

plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred_linear,label='Linear',alpha=0.6)
plt.scatter(y_test,y_pred_poly,label='Polynomial (degree=2)',alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--',label='Perfect Prediction')
plt.title("Linear vs Polynomial Prediction")
plt.xlabel("Actual Price ")
plt.ylabel("Predicted Price ")
plt.legend()
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="924" height="535" alt="image" src="https://github.com/user-attachments/assets/e8b1d2c0-a99f-4298-a879-5e56535bff6c" />
<img width="1008" height="469" alt="image" src="https://github.com/user-attachments/assets/e958b74f-7020-4b20-a780-fde0e4dc6d92" />


## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
