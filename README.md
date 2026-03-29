# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program.

Step 2: Import required libraries:

pandas, numpy for data handling
sklearn modules for model building and evaluation
matplotlib for visualization

Step 3: Load the dataset (encoded_car_data.csv) using pandas.read_csv().

Step 4: Display the first few rows of the dataset using head() to understand the data.

Step 5: Select input features (independent variables):

horsepower
enginesize
citympg
highwaympg

Step 6: Define the target variable (dependent variable):

price

Step 7: Split the dataset into training and testing sets using train_test_split() with 80% training and 20% testing data.

Step 8: Build Linear Regression model using Pipeline:

Apply StandardScaler for feature scaling
Apply LinearRegression model
Train the model using training data

Step 9: Predict car prices on test data using the Linear model.

Step 10: Evaluate Linear Regression model using:

Mean Squared Error (MSE)
R² Score

Step 11: Build Polynomial Regression model using Pipeline:

Apply PolynomialFeatures (degree = 2)
Apply StandardScaler
Apply LinearRegression model
Train the model

Step 12: Predict car prices on test data using the Polynomial model.

Step 13: Evaluate Polynomial Regression model using:

Mean Squared Error (MSE)
R² Score

Step 14: Display the results including Name and Register Number.

Step 15: Visualize the results:

Plot Actual vs Predicted prices for both models
Draw reference line for perfect prediction

Step 16: Compare both models based on performance metrics.

Step 17: Stop the program.

## Program:
```
/*
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: R JAYENTHAN
RegisterNumber: 25011312    //   212225240057
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


print("Name: R JAYENTHAN ")
print("Reg No: 212225240057 ")
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
![WhatsApp Image 2026-03-30 at 5 15 52 AM](https://github.com/user-attachments/assets/051012ef-71fb-49ab-944c-1e8baf060342)

<img width="1008" height="469" alt="image" src="https://github.com/user-attachments/assets/bb0bdc49-6880-4858-821b-954a24baa549" />



## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
