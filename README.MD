# DataMagic

Preprocesses your data to make it ready for analysis.  System can work with user inputs, or allow DataMagic to do the hardwork for you, and just click go.

----------------------------------------

## First Order Logic

If less an x percentage of rows are corrupt, then those rows are dropped.  This is the easy route.

## Second Order Logic

If the easiest fix doesn't work, then we have to look at the different types of data.

It's important to know how this data will be used.

### First Is It Categorical Data for Supervised Learning

- Assume last column
- Identify any column that could potentially be it.



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

### Supervised Learning

Right now it runs through about ten different standard ML classifiers.
