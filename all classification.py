# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:31:40 2018

@author: ade putra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#==============================================================================
#import dataset
#==============================================================================
df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, [2,3]].values
y = df.iloc[:, 4].values

#==============================================================================
##encoder categorical data
#==============================================================================
#from sklearn.preprocessing import LabelEncoder
#LE = LabelEncoder()
#X[:, 0] = LE.fit_transform(X[:, 0])

#==============================================================================
#train test split
#==============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
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
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors= 8,p=2),
        "NearestCentroid": NearestCentroid(),
        "MLPClassifier": MLPClassifier(solver = 'adam', hidden_layer_sizes = 300, 
                                       random_state = 0), 
        #hidden layer paling bagus 75% dari data
        "DecisionTreeClassifier": DecisionTreeClassifier(criterion = 'entropy',
                                                         random_state = 0),
        "RandomForestClassifier": RandomForestClassifier(random_state = 0, 
                                                         n_estimators = 300),
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

#visualizing data
from matplotlib.colors import ListedColormap
X_set, y_set = X_train,y_train
#X_set_test, y_set_test = X_test,y_test
def grid(X_set,stp):
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                                   stop = X_set[:, 0].max() + 1, step = stp),
                     np.arange(start = X_set[:, 1].min() - 1, 
                               stop = X_set[:, 1].max() + 1, step = stp))
    return X1,X2

def plot_contour(X1,X2,X_set,y_set,dict_list):
    for name, estimator in dict_list:
        print('\nestimator = {}'.format(name))
        plt.figure()
        plt.title('{} (Training set)'.format(name))
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.contourf(X1, X2, 
                     estimator.predict(np.array([X1.ravel(), 
                                                 X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.legend()

X1, X2 = grid(X_set,0.01)
#X1_test,X2_test = grid(X_set_test,0.01)
plot_contour(X1,X2,X_set,y_set,ESTIMATORS.items())
#plot_contour(X1_test,X2_test,X_set_test,y_set_test,ESTIMATORS.items())
plt.show()