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
import supervise
import data

# Inputs
dataset, classcolumn, headers, folds = sys.argv[1:]
headers = None if headers == "None" else headers
classcolumn = int(classcolumn) if headers == None else classcolumn
folds = int(folds)

# Get data and create column and data sets
data, class_data, class_column = supervise.create_column_class(dataset,classcolumn,headers)

# Create Cross-validation
accuracies, classifiers = supervise.multiclass(folds,class_data,class_column)

for clf_name, classifier in classifiers:
    print('average_accuracy for classifier ',clf_name.upper(),' with', folds, "folds is:",accuracies[clf_name]['average']," with a standard deviation of:",accuracies[clf_name]['stdev'])