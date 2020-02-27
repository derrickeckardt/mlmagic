import unittest
import data
import pandas as pd

class test_data(unittest.TestCase):

    # Read in test Datafile
    def setUp(self):
        self.dataset = 'iris.data'
        self.headers = None
        self.classcolumn = 4
        self.folds = 2

    def test_create_column_class(self):
        self.data, self.class_data, self.class_column = data.create_column_class(self.dataset, self.classcolumn, self.headers)
        try:
            class_data_width = self.class_data.shape[1]
        except:
            class_data_width = 1
        try:
            class_column_width = self.class_column.shape[1]
        except:
            class_column_width = 1
        self.assertEqual(self.data.shape[1], class_column_width+class_data_width)
    
    def test_get_missing_values(self):
        self.assertTrue(True)
    
    def test_basic_clean_data(self):
        self.data, self.class_data, self.class_column = data.create_column_class(self.dataset, self.classcolumn, self.headers)
        self.data = data.basic_clean_data(self.data)
        self.assertEqual(self.data.isna().sum().sum(),0)
        
if __name__ == '__main__':
    unittest.main()