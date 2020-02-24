#!/usr/bin/env python3
import pandas as pd
import numpy as np

# create data sets
def create_column_class(dataset, classcolumn, headers):
    # Read in Datafile
    data = pd.read_csv(dataset, header=headers)
    data = clean_data(data)
    class_column = data[classcolumn]
    class_data = data.drop(classcolumn,1)
    return data, class_data, class_column

def clean_data(data):
    return data