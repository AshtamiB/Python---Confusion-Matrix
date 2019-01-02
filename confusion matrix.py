# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 06:21:41 2018

@author: Ashtami
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix #(y_actual, y_predicted)
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Loading Dataset --------------------------
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features.
y = iris.target

# ------------- Create the SVM/NB/RF object --------------------------#
RF = RandomForestClassifier(n_estimators=3, random_state=12)
RF.fit(X, y)
Z_RF = RF.predict(X)

NB = GaussianNB()
NB.fit(X, y)
Z_NB = NB.predict(X)

C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C)
svc.fit(X, y)
Z_SV = svc.predict(X)
# -------------------- CONFUSION MATRIX -----------------------#
CM_RF = confusion_matrix(y, Z_RF)
CM_NB = confusion_matrix(y, Z_NB)
CM_SV = confusion_matrix(y, Z_SV)
print(CM_RF)
print('----------------------------')
print(CM_NB)
print('----------------------------')
print(CM_SV)
plt.subplot(131)
sns.heatmap(CM_RF.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.subplot(132)
sns.heatmap(CM_NB.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.subplot(133)
sns.heatmap(CM_SV.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()