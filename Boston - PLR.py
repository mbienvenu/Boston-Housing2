# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 19:27:33 2018

@author: cecil
"""


#DAta processing

import numpy as np
import pandas as pd
import os
import csv
import sys
import re


boston_train = pd.read_csv("train.csv")
boston_test = pd.read_csv("test.csv")


X_train=boston_train.iloc[:,1:14].values 
y_train=boston_train.iloc[:,14].values

X_test = boston_test.iloc[:,1:14].values 

np.set_printoptions(threshold=100) 

# No need to apply feature scaling as the Library takes care of it
#Model using Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree=10)
x_poly = poly_reg.fit_transform(X_train)
line_reg = LinearRegression()
line_reg.fit(x_poly,y_train)


#predicting the result using polynomial regression
y_pred = line_reg.predict(poly_reg.fit_transform(X_test))
#write_csv(boston_test['ID'],ypred,'medv_submission_PLR.csv')



#df = pd.DataFrame(y_pred)
#First write version to file as a whole array in a single line 
with open("medv_submission_PLR.csv","w") as outfile:
    writer = csv.writer(outfile)
    row = y_pred
    writer.writerow(row)
    
outfile.close()

 

