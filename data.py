#!/usr/bin/env python3
import pandas as pd
import numpy as np

# create data sets
def create_column_class(dataset, classcolumn, headers):
    
    # Read in Datafile
    missing_values = get_missing_values()
    data = pd.read_csv(dataset, header=headers, na_values=missing_values)
    data = basic_clean_data(data)
    class_column = data[classcolumn]
    class_data = data.drop(classcolumn,1)
    return data, class_data, class_column

def get_missing_values():
    # Eventually add ability to determine which missing values
    missing_values = ["n/a", "N/A","N/a","n/A","na","NA","Na","nA","NAN","-"]
    return missing_values

def basic_clean_data(data):
    # clear blanks, empty strings, NaN, N/A
    data = data.applymap(lambda x: np.nan if isinstance(x, str) and (not x or x.isspace()) else x)
    # does not change 'Na' or 'NAN' or -
    print(data.head())
    
    # issue warning about other ways it will not catch
    
    # Null, NaN, None
    # Since data already loaded
    
    # sparse columns
        # dropping columns with na
    # Missing rows
        # using the mode or the mean
    
    # Look for data that is out of place, a number when all th other respons are yes no
    
    # Extreme outliers
    missing_values = data.isna().sum().sum()
    print(missing_values)
    
    ## advanced cleaning
    # Address formatting
    # data formatting
    # spell checking
    
    
    
    
    
    return data