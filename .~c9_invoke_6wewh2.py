import unittest
import pandas as pd
import argparse
import sys


from supervise import multiclass

class test_multiclass(unittest.TestCase):

    # Read in test Datafile
    data = pd.read_csv('iris.data', header=None)
    folds = 5
    class_column = data[4]
    class_data = data.drop(4,1)
    print(class_column.head())

    def test_accuracies(self, folds,class_data,class_column):

        accuracies, classifiers = multiclass(folds,class_data,class_column)
        for clf_name, classifier in classifiers:
            stdev = True if accuracies[clf_name]['stdev'] <= 1 and accuracies[clf_name]['stdev'] >= 0 else False
            self.assertTrue(stdev)
            average = True if accuracies[clf_name]['average'] <= 1 and accuracies[clf_name]['average'] >= 0 else False
            self.assertTrue(average)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='iris.data')
    parser.add_argument('classcolumn', default='4')
    parser.add_argument('headers', default='None')
    parser.add_argument('folds', default='5')
    
    
    args = parser.parse_args()
    print(args)
    # TODO: Go do something with args.input and args.filename
    
    # # Now set the sys.argv to the unittest_args (leaving sys.argv[0] alone)
    sys.argv[1] = args.dataset
    sys.argv[2] = args.classcolumn
    sys.argv[3] = args.headers
    sys.argv[4] = args.folds

    unittest.main()