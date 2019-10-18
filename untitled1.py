# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:15:52 2019

@author: sridhar
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


from sklearn.cross_validation import train_test_split
X_train, X_test ,y_train ,y_test = train_test_split(X ,y , test_size=0.2, random_state=0 )


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
#y_pred = lin_reg.predict(X_test)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly , y)

#Visulizing Linear Regression Model
plt.scatter(X , y , color='red')
plt.plot(X,lin_reg.predict(X), color='blue')
#Visulizing Polynominal Regression Model
plt.scatter(X , y , color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
#predict the result from linear regression 
lin_reg.predict(6.5)
#Predicting Result From the Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))
