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
    missing_values = ["n/a", "N/A","N/a","n/A","na","NA","Na","nA","NAN","-","", " ", "  "]
    # issue warning about other ways it will not catch
    return missing_values

def drop_sparse_columns():
    
    return 

def basic_clean_data(data):
    # First, identify how many
    row_drop_threshold = 0.01
    row_na_count = data.isnull().any(axis=1).sum()
    na_values = data.isna().sum()
    data_shape = data.shape

    if row_na_count <= data_shape[0]*row_drop_threshold:
        # just drop the rows
        print('Dropping rows with values that are NaN')
        data = data.dropna()
        # Option could be to just fill them with the mode
        print('Data successfully cleaned')
    else:
        # we can't just drop the rows
        print("Changing all NaN values to modes")
        for column in data.columns:
            data[column].fillna(data[column].mode()[0], inplace=True)

        print(data.head())


    
    # for column, na_count in zip(data.columns, na_values):
    #     if na_count < data_shape[0]*threshold:
    #         print('oops')

    #### Things to Check
    # sparse columns
        # dropping columns with na
    # Missing rows
        # using the mode or the mean
    # Look for data that is out of place, a number when all th other respons are yes no
    
    # Extreme outliers
    

    
    ##### advanced cleaning
    # Address formatting
    # data formatting
    # data formatting
    # spell checking
    
    return data