#!/usr/bin/env python3
import sys
from statistics import stdev, mean
import cProfile
import supervise
import data

# Inputs
dataset, classcolumn, headers, folds = sys.argv[1:]
headers = None if headers == "None" else 0
classcolumn = int(classcolumn) if headers == None else classcolumn
folds = int(folds)

# Get data and create column and data sets
data, class_data, class_column = data.create_column_class(dataset,classcolumn,headers)

# Create Cross-validation
accuracies, classifiers = supervise.multiclass(folds,class_data,class_column)

for clf_name, classifier in classifiers:
    print('average_accuracy for classifier ',clf_name.upper(),' with', folds, "folds is:",accuracies[clf_name]['average']," with a standard deviation of:",accuracies[clf_name]['stdev'])