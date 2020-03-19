#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# create data sets
def create_column_class(dataset, classcolumn, headers, index_column):
    
    # Read in Datafile
    missing_values = get_missing_values()
    raw_data = pd.read_csv(dataset, header=headers, na_values=missing_values, index_col=index_column)
    data = basic_clean_data(raw_data,classcolumn)
    class_column = data[classcolumn]
    class_data = data.drop(classcolumn,1)
    return data, class_data, class_column

def get_missing_values():
    # Eventually add ability to determine which missing values
    missing_values = ["n/a", "N/A","N/a","n/A","na","NA","Na","nA","NAN","-","", " ", "  "]
    # issue warning about other ways it will not catch
    return missing_values

def drop_sparse_columns(data, row_count, classcolumn, sparse_column_threshold):
    for column, column_na_value in zip(data.columns,data.isna().sum()):
        if column_na_value / row_count > sparse_column_threshold and column != classcolumn :
            print("Column '",column,"'has ",column_na_value/row_count," as NaN.  Do you want to drop it? (Y/N)")
            drop_column_input = input() #commented out for testing
            print(drop_column_input)
            if drop_column_input.upper() == "Y":
                # data = data.drop(columns=column)
                print("drop",column)
            elif drop_column_input.upper() == "N":
                print("Would you like blanks to be the mode, 0, or None")
                blank_input = input()
                if blank_input.lower() == "mode":
                    data = replace_with_mode(data)
                elif blank_input.lower() == "0":
                    data = replace_with(data, 0)
                elif blank_input.lower() == "none":
                    data = replace_with(data, "none")
                else:
                    #other
                    data = replace_with(data,blank_input.lower())
    return data

def replace_with_mode(data):
    for column in data.columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    return data
    
def replace_with(data,value):
    for column in data.columns:
        data[column].fillna(value, inplace=True)
    return data

# current system is too simplistic, but it's a start.
def basic_clean_data(data, classcolumn):
    # presets - make an option in production
    sparse_column_threshold = 0.01
    row_drop_threshold = 0.05
    row_na_count = data.isna().any(axis=1).sum()
    na_values = data.isna().sum()
    row_count = data.shape[0]

    if row_na_count <= row_count*row_drop_threshold:
        # just drop the rows
        print('Dropping rows with values that are NaN')
        data = data.dropna()
    else:
        # we can't just drop the rows
        # Removing sparse columns
        print('Removing sparse columns')
        data = drop_sparse_columns(data, row_count, classcolumn, sparse_column_threshold)
        print('Data successfully cleaned')

    # preprocessing data by encoding it
    # data = OneHotEncoder().fit_transform(data)

    # Documentation Reference:
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html

    #### Things to Handle Better
    # sparse columns
        # dropping columns with a lot of na
        # ensure you always have at least two columns
    # sparse row
        # drop row if too many values
    # after doing this, then recheck for threshold to see if it works then.        

    #### Intelligent Items
    # Look for data that is mislabeled, ex a number when should be yes/no
    # Extreme outliers, when the orders of magnitude are off 
        # there are some different option PyOD, but htat requires keras and tensorflow
        # https://pyod.readthedocs.io/en/latest/
        # sklearn has some outlier detection options
        # https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html#sphx-glr-auto-examples-plot-anomaly-comparison-py
    
    ##### advanced cleaning
    # Address formatting
    # data formatting
    # date formatting
    # spell checking
    
    return data