# Need to pip install xlrd to use pd.read_excel
# data is from: 
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read from excel file
df = pd.read_excel('mlr02.xls')
X = df.values

# using age to predict systolic blood pressure
# X[:,1] means that we are selecting the X2 column (age)
plt.scatter(X[:,1], X[:,0])
plt.show()
# looks pretty linear! :o

# using weight to predict systolic blood pressure
# X[:,2] means that we are selecting the X3 column (weight in pounds)
plt.scatter(X[:,2], X[:,0])
plt.show()
# looks pretty linear! :o

#  Add the bias term x0 = 1
df['ones'] = 1
Y = df['X1']
X = df[['X2', 'X3', 'ones']]
X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

# Get R^2 (R Squared)
def get_r2(X, Y):
    w = np.linalg.solve( X.T.dot(X), X.T.dot(Y) )
    Yhat = X.dot(w)

    # determine how good the model is by computing the r-squared
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    return r2

print("r2 for x2 only:", get_r2(X2only, Y))
print("r2 for x3 only:", get_r2(X3only, Y))
print("r2 for both:", get_r2(X, Y))
