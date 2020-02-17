#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

dataset, classcolumn, headers = sys.argv[1:]
headers = None if headers == "None" else headers
classcolumn = int(classcolumn) if headers == None else classcolumn

data = pd.read_csv(dataset, header=headers)
class_column = data[classcolumn]
class_data = data.drop(classcolumn,1)

for i in range(1,10):
    x_train, x_test, y_train, y_test = train_test_split(class_data, class_column, test_size = i/10)
    
    # Decision Trees
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train,y_train)
    predictions = clf.predict(x_test)
    print('test size:',i/10,'accuracy:',accuracy_score(y_test,predictions))