#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from statistics import stdev, mean
import cProfile

# Inputs
dataset, classcolumn, headers, folds = sys.argv[1:]
headers = None if headers == "None" else headers
classcolumn = int(classcolumn) if headers == None else classcolumn
folds = int(folds)
neighbors = 3

# Read in Datafile
data = pd.read_csv(dataset, header=headers)
class_column = data[classcolumn]
class_data = data.drop(classcolumn,1)

# Create Cross-validation
kf = KFold(n_splits=folds, shuffle=True)
classifiers = [('Decision Tree',DecisionTreeClassifier(max_depth=5)),
                ('kNN',KNeighborsClassifier(n_neighbors=neighbors)),
                ('Support Vector Linear', SVC(kernel="linear", C=0.025)),
                ('Support Vector Radial', SVC(gamma=2, C=1)),
                ('Gaussian',GaussianProcessClassifier(1.0 * RBF(1.0))),
                ('Random Forest',RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
                ('Neural Net',MLPClassifier(alpha=1, max_iter=1000)),
                ('AdaBoost',AdaBoostClassifier()),
                ('Naive Bayes',GaussianNB()),
                ('Quadratic Discrimation',QuadraticDiscriminantAnalysis())
                ]


def multiclass(kf,class_data,class_column,classifiers):

    accuracies = {}

    for clf_name, classifier in classifiers:
        accuracies[clf_name] = {}
        accuracies[clf_name]['total'] = []
    
    for train_index, test_index in kf.split(class_data):
        # x_train, x_test, y_train, y_test = train_test_split(class_data, class_column, test_size = i/10)
        x_train, y_train = class_data.iloc[train_index], class_column.iloc[train_index]
        x_test, y_test = class_data.iloc[test_index], class_column.iloc[test_index]
    
        # Cycle through all the classifiers, for this dataset
        for clf_name, classifier in classifiers:
            clf = classifier
            clf = clf.fit(x_train,y_train)
            predictions = clf.predict(x_test)
            accuracy = accuracy_score(y_test,predictions)
            accuracies[clf_name]['total'].append(accuracy)
        
    for clf_name, classifier in classifiers:
        accuracies[clf_name]['average'] = mean(accuracies[clf_name]['total'])
        accuracies[clf_name]['stdev'] = stdev(accuracies[clf_name]['total'])
    
    return accuracies

accuracies = multiclass(kf,class_data,class_column,classifiers)
for clf_name, classifier in classifiers:
    print('average_accuracy for classifier ',clf_name.upper(),' with',folds,"folds is:",accuracies[clf_name]['average']," with a standard deviation of:",accuracies[clf_name]['stdev'])