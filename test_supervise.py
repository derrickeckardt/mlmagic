import unittest
import pandas as pd
import supervise

class test_multiclass(unittest.TestCase):

    # Read in test Datafile
    def setUp(self):
        self.dataset = 'iris.data'
        self.headers = None
        self.classcolumn = 4
        self.folds = 2
        self.data, self.class_data, self.class_column = supervise.create_column_class(self.dataset, self.classcolumn, self.headers)
        # print(self.data.head())

    def test_create_column_class(self):
        try:
            class_data_width = self.class_data.shape[1]
        except:
            class_data_width = 1
        try:
            class_column_width = self.class_column.shape[1]
        except:
            class_column_width = 1
        print(class_column_width, class_data_width)
        self.assertEqual(self.data.shape[1], class_column_width+class_data_width)

    def test_multiclass(self):
        accuracies, classifiers = supervise.multiclass(self.folds,self.class_data,self.class_column)
        for clf_name, classifier in classifiers:
            stdev = True if accuracies[clf_name]['stdev'] <= 1 and accuracies[clf_name]['stdev'] >= 0 else False
            self.assertTrue(stdev)
            average = True if accuracies[clf_name]['average'] <= 1 and accuracies[clf_name]['average'] >= 0 else False
            self.assertTrue(average)



if __name__ == '__main__':
    unittest.main()