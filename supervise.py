#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


dataset, classcolumn, headers, folds = sys.argv[1:]
headers = None if headers == "None" else headers
classcolumn = int(classcolumn) if headers == None else classcolumn
folds = int(folds)

data = pd.read_csv(dataset, header=headers)
class_column = data[classcolumn]
class_data = data.drop(classcolumn,1)

# folds = 50
kf = KFold(n_splits=folds, shuffle=True)
count = 1
accuracy_total = 0
for train_index, test_index in kf.split(class_data):
    # print("TRAIN:", train_index, "TEST:", test_index)

    # x_train, x_test, y_train, y_test = train_test_split(class_data, class_column, test_size = i/10)
    x_train, y_train = class_data.iloc[train_index], class_column.iloc[train_index]
    x_test, y_test = class_data.iloc[test_index], class_column.iloc[test_index]

    # Decision Trees
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train,y_train)
    predictions = clf.predict(x_test)
    accuracy = accuracy_score(y_test,predictions)
    print('fold:',count,'accuracy:',accuracy)
    count +=1 
    accuracy_total += accuracy
print("average_accuracy for",folds,"folds is:",accuracy_total/folds)