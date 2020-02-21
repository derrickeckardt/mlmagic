import unittest
import pandas as pd
import argparse
import sys


from supervise import multiclass

class test_multiclass(unittest.TestCase):

    # Read in test Datafile
    def setUp(self):
        self.data = pd.read_csv('iris.data', header=None)
        self.folds = 2
        self.class_column = self.data[4]
        self.class_data = self.data.drop(4,1)

    def test_multiclass(self):
        accuracies, classifiers = multiclass(self.folds,self.class_data,self.class_column)
        for clf_name, classifier in classifiers:
            stdev = True if accuracies[clf_name]['stdev'] <= 1 and accuracies[clf_name]['stdev'] >= 0 else False
            self.assertTrue(stdev)
            average = True if accuracies[clf_name]['average'] <= 1 and accuracies[clf_name]['average'] >= 0 else False
            self.assertTrue(average)

if __name__ == '__main__':
    unittest.main()