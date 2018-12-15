# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:31:40 2018

@author: ade putra
"""

import numpy as np
import pandas as pd

#==============================================================================
#import dataset
#==============================================================================
#df = pd.read_csv('Social_Network_Ads.csv')
#X = df.iloc[:, [2,3]].values
#y = df.iloc[:, 4].values

df2 = pd.read_csv('bank.csv',sep=';')
X = np.array(df2.drop(['marital','education','default','y','contact','balance',
                       'month','day','campaign','job','duration','poutcome','pdays'],1))
y = np.array(df2['y'])

#==============================================================================
##encode categorical data
#==============================================================================
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
LEy = LabelEncoder()
X[:, 1] = LE.fit_transform(X[:, 1])
X[:, 2] = LE.fit_transform(X[:, 2])
y = LEy.fit_transform(y)

#==============================================================================
#train test split
#==============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

#==============================================================================
# Feature Scaling
#==============================================================================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#==============================================================================
#make model 
#==============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

ESTIMATORS = {
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors= 10),
        "NearestCentroid": NearestCentroid(),
        "MLPClassifier": MLPClassifier(solver = 'adam', hidden_layer_sizes = 2712, 
                                       random_state = 0), 
        #hidden layer paling bagus 75% dari data
        "DecisionTreeClassifier": DecisionTreeClassifier(criterion = 'entropy',
                                                         random_state = 0),
        "RandomForestClassifier": RandomForestClassifier(random_state = 0, 
                                                         n_estimators = 2712),
        "GaussianNB": GaussianNB(),
        "SVC": SVC(kernel='rbf',decision_function_shape='ovo'),
        "LogisticRegression": LogisticRegression(solver='lbfgs',
                                                 random_state = 0),
        }

#evaluation and making confusion matrix
from sklearn.metrics import confusion_matrix 
y_pred = dict()
score = dict()
cm = dict()
for name, estimator in ESTIMATORS.items():
    print('\nestimator = {}'.format(name))
    estimator.fit(X_train, y_train)         # fit() with instantiated object
    y_pred[name] = estimator.predict(X_test)
    score[name] = estimator.score(X_test,y_test)
    cm[name] = confusion_matrix(y_test,y_pred[name])
