import unittest
import pandas as pd
import supervise
import data

class test_supervise(unittest.TestCase):

    # Read in test Datafile
    def setUp(self):
        self.dataset = 'iris.data'
        self.headers = None
        self.classcolumn = 4
        self.folds = 2
        self.data, self.class_data, self.class_column = data.create_column_class(self.dataset, self.classcolumn, self.headers)

    def test_multiclass(self):
        accuracies, classifiers = supervise.multiclass(self.folds,self.class_data,self.class_column)
        for clf_name, classifier in classifiers:
            stdev = True if accuracies[clf_name]['stdev'] <= 1 and accuracies[clf_name]['stdev'] >= 0 else False
            self.assertTrue(stdev)
            average = True if accuracies[clf_name]['average'] <= 1 and accuracies[clf_name]['average'] >= 0 else False
            self.assertTrue(average)

if __name__ == '__main__':
    unittest.main()