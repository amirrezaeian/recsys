
import unittest
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from lightfm import LightFM

from src.metric_utilities import model_metrics


class TestModelMetrics(unittest.TestCase):
    """Test model_metrics method"""
    def setUp(self):
        self.train_matrix = coo_matrix(np.array([[1, 2, 3], [4, 5, 6]]))
        self.test_matrix = coo_matrix(np.array([[1, 2, 3], [4, 5, 6]]))
        self.model = LightFM()
        self.k = 10
        self.no_epochs = 10

    def test_model_metrics(self):
        df_actual = model_metrics(self.model,
                                  self.train_matrix,
                                  self.test_matrix,
                                  self.k,
                                  self.no_epochs,
                                  True)
        self.assertIsInstance(df_actual, pd.DataFrame)
        self.assertTupleEqual(df_actual.shape, (3, 3))
        expected_cols = ['Evaluation Metric', 'Train', 'Test']
        self.assertListEqual(list(df_actual.columns), expected_cols)
        expected_rows = ['Average AUC', 'Average Precision@{}'.format(self.k), 'Average recall@{}'.format(self.k)]
        self.assertListEqual(list(df_actual['Evaluation Metric'].values), expected_rows)
        #todo:  Add more unit testing for this method in the upcoming PR.
