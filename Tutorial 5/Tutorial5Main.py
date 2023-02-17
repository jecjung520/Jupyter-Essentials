# Tutorial5Main.py

# Load package
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import math
# Io for loading .mat file
import scipy.io as io

# Load dataset
matr=io.loadmat('FullSet-wdbc.mat')
X=matr['features']
print(X.shape)
Y=matr['BinaryLabel']
print(Y.shape)
y=Y[:,0]
print(y.shape)

# Students are requested to partition the first 90% of data as training data 
# and the rest as testing data

# <Students fill in codes here>

# Create classifier
# More information about LogisticRegression is available at:
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
clf=LogisticRegression(max_iter=100000)

# Train and evaluation
clf.fit(X_train, y_train)

# Decision function is w*x + w_0,
# where w can be extracted by clf.coef_
# and w_0 can be extracted by clf.intercept_

Y_df = clf.decision_function(X_test)
print("Decision Function (w*x + w_0) = ")
print(Y_df)

# You can try this to validate
print("clf.coef_ (w) =")
print(clf.coef_)
print("clf.intercept (w_0) =")
print(clf.intercept_)

# This should be the decision function w*x + w_0 corresponding to the first testing sample
DF_first_test_sample = np.dot(X_test[0,:],clf.coef_[0,:])+clf.intercept_
print("DF_first_test_sample =")
print(DF_first_test_sample)

# This provides the probability for the '0' class followed by the '1' class for all test samples
Y_pred_proba = clf.predict_proba(X_test)
print("Y_pred_proba=")
print(Y_pred_proba)

# You can try this to validate
# The following should provide the posterior probability of the '1' class for the first testing sample
sigma_first_test_sample = 1/(1+math.exp(-DF_first_test_sample))
print("sigma_first_test_sample=")
print(sigma_first_test_sample)

# This provides the predicted class for all test samples
Y_pred=clf.predict(X_test)
print("Y_pred=")
print(Y_pred)

print("y_test=")
print(y_test)

# This computes the accuracy of classification
acc=accuracy_score(y_test, Y_pred)
print('Accuracy=')
print(acc)