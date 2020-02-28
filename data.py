#!/usr/bin/env python3
import pandas as pd
import numpy as np

# create data sets
def create_column_class(dataset, classcolumn, headers):
    
    # Read in Datafile
    missing_values = get_missing_values()
    raw_data = pd.read_csv(dataset, header=headers, na_values=missing_values)
    data = basic_clean_data(raw_data)
    class_column = data[classcolumn]
    class_data = data.drop(classcolumn,1)
    return data, class_data, class_column

def get_missing_values():
    # Eventually add ability to determine which missing values
    missing_values = ["n/a", "N/A","N/a","n/A","na","NA","Na","nA","NAN","-",""]
    # issue warning about other ways it will not catch
    return missing_values

def drop_sparse_columns():
    
    return 

def basic_clean_data(data):
    # First, identify how many
    threshold = 0.05
    na_values = data.isna().sum()
    data_shape = data.shape()
    print(data_shape)
    
    for column, na_count in zip(data.columns, na_values):
        if na_count < data_shape[0]*threshold

    # sparse columns
        # dropping columns with na
    # Missing rows
        # using the mode or the mean
    
    # Look for data that is out of place, a number when all th other respons are yes no
    
    # Extreme outliers
    

    
    print(na_values)
    
    
    ## advanced cleaning
    # Address formatting
    # data formatting
    # spell checking
    
    
    
    
    
    return data