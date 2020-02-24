import unittest
import data

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
        
        
if __name__ == '__main__':
    unittest.main()