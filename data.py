#!/usr/bin/env python3
import pandas as pd
import numpy as np

# create data sets
def create_column_class(dataset, classcolumn, headers):
    
    # Read in Datafile
    missing_values = get_missing_values()
    raw_data = pd.read_csv(dataset, header=headers, na_values=missing_values)
    data = basic_clean_data(raw_data,classcolumn)
    class_column = data[classcolumn]
    class_data = data.drop(classcolumn,1)
    return data, class_data, class_column

def get_missing_values():
    # Eventually add ability to determine which missing values
    missing_values = ["n/a", "N/A","N/a","n/A","na","NA","Na","nA","NAN","-","", " ", "  "]
    # issue warning about other ways it will not catch
    return missing_values

def drop_sparse_columns(data, row_count, classcolumn):
    sparse_column_threshold = 0.10
    for column, column_na_value in zip(data.columns,data.isna().sum()):
        if column_na_value / row_count > sparse_column_threshold and column != classcolumn :
            data = data.drop(columns=column)
    return data

def replace_with_mode(data):
    for column in data.columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    return data

# current system is too simplistic, but it's a start.
def basic_clean_data(data, classcolumn):
    # First, identify how many
    row_drop_threshold = 0.25
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
        data = drop_sparse_columns(data, row_count, classcolumn)
        print("Changing all remaining NaN values to modes")
        data = replace_with_mode(data)
        print('Data successfully cleaned')

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