# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 22:56:16 2018

@author: cecil
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 21:41:10 2018

@author: cecil
"""



#DAta processing

import numpy as np
import pandas as pd
import csv



boston_train = pd.read_csv("train.csv")
boston_test = pd.read_csv("test.csv")


X_train=boston_train.iloc[:,1:14].values 
y_train=boston_train.iloc[:,14].values

X_test = boston_test.iloc[:,1:14].values 

#Fitting Decision Tree to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=212)
regressor.fit(X_train,y_train)


#Building the predictions 
y_pred = regressor.predict(X_test)


with open("medv_submission_DTree.csv","w") as outfile:
    writer = csv.writer(outfile,delimiter = "\n")
    row = y_pred
    writer.writerow(row)

    
outfile.close()




