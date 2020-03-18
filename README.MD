# ML Magic

ML Magic is a plug-and-play AI interface for setting up AI workflows.

## To run

In order to run, use this command:

    time ./index.py dataset classcolumn headers index_column folds
    time ./index.py data/iris.data 4 None None 5
    time ./index.py data/patient.csv state Yes 'patient_id' 5
    
 
where dataset is the dataset to use, classcolumn is the column to use for classification, and headers indicates whether there are column titles.

## To test

In order to test, below are a list of all the tests:

    python3 -m unittest test_supervise
    python3 -m unittest test_data


## About the Code

The codebase is built on python, and will ultimately be deployed into Django.

### Data Clean-up and Preprocessing

The hardest part of any data prep.  ML Magic makes it easy, by making decisions for the user, autodetecting columns and making decisions for the user.  Of course, the user can do more work if you want it to, but why would you?

### Supervise Learning

