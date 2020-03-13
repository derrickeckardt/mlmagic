# ML Magic

Allows the user to run multiple machine learning algorithms from the command line.

## To run

In order to run, use this command:

    ./index.py dataset classcolumn headers folds
    ./index.py data/iris.data 4 None 5
    ./index.py data/patient.csv state Yes 5
    
 
where dataset is the dataset to use, classcolumn is the column to use for classification, and headers indicates whether there are column titles.

## To test

In order to test, below are a list of all the tests:

    python3 -m unittest test_supervise
    python3 -m unittest test_data