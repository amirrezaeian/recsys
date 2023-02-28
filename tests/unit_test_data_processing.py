import pytest
import pandas as pd
from scipy.sparse import coo_matrix
from pandas._testing import assert_frame_equal
from unittest import TestCase
from src.data_processing_utilities import (build_interaction_matrix, build_user_dict,
                                           build_item_dict, data_preprocessing_implicit_feedback,
                                           create_train_test_data)


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    """
        Build a data for testing methods.
    """
    data = pd.DataFrame([
        {'userId': 1, 'movieId': '31', 'rating': 2, 'timestamp': 1260759179},
        {'userId': 1, 'movieId': '1029', 'rating': 3.0, 'timestamp': 1260759180},
        {'userId': 1, 'movieId': '1029', 'rating': 3.0, 'timestamp': 1260759190},
        {'userId': 2, 'movieId': '1129', 'rating': 2.0, 'timestamp': 1260759190},
        {'userId': 2, 'movieId': '1172', 'rating': 4.0, 'timestamp': 1260759190},
    ])
    return data


def test_build_interaction_matrix(df):
    """
    Test build_interaction_matrix method.
    """

    df_expected = pd.DataFrame()
    df_expected.index.name = "userId"
    df_expected.columns.name = "movieId"
    df_expected.loc[1, '31'] = 2.0
    df_expected.loc[1, '1029'] = 6.0
    df_expected.loc[1, '1129'] = 0.0
    df_expected.loc[1, '1172'] = 0.0

    df_expected.loc[2, '31'] = 0.0
    df_expected.loc[2, '1029'] = 0.0
    df_expected.loc[2, '1129'] = 2.0
    df_expected.loc[2, '1172'] = 4.0

    df_out_built = build_interaction_matrix(df, "userId", "movieId", "rating", False)
    df_out_built_with_norm = build_interaction_matrix(df, "userId", "movieId", "rating",
                                                      norm=True, threshold=2)

    df_with_norm_expected = pd.DataFrame()
    df_with_norm_expected.index.name = "userId"
    df_with_norm_expected.columns.name = "movieId"
    df_with_norm_expected.loc[1, '31'] = 1.0
    df_with_norm_expected.loc[1, '1029'] = 1.0
    df_with_norm_expected.loc[1, '1129'] = 0.0
    df_with_norm_expected.loc[1, '1172'] = 0.0

    df_with_norm_expected.loc[2, '31'] = 0.0
    df_with_norm_expected.loc[2, '1029'] = 0.0
    df_with_norm_expected.loc[2, '1129'] = 1.0
    df_with_norm_expected.loc[2, '1172'] = 1.0

    assert df_out_built.shape == (df.userId.nunique(), df.movieId.nunique())
    assert df_out_built_with_norm.shape == (df.userId.nunique(), df.movieId.nunique())
    assert_frame_equal(df_out_built.sort_index(axis=1), df_expected.sort_index(axis=1))
    assert_frame_equal(df_out_built_with_norm.sort_index(axis=1), df_with_norm_expected.sort_index(axis=1))


def test_build_user_dict(df):
    """
    Test build_user_dict method.
    """

    df_out_built = build_interaction_matrix(df, "userId", "movieId", "rating", False)

    actual_dict = build_user_dict(df_out_built)
    expected_dict = {1: 0, 2: 1}
    TestCase().assertDictEqual(expected_dict, actual_dict)


def test_build_item_dict(df):
    """
    Test build_item_dict method.
    """

    data = pd.DataFrame([
        {'movieId': '31', 'title': 'Toy Story', 'timestamp': 1260759179},
        {'movieId': '1029', 'title': 'Father of the Bride Part II (1995)', 'timestamp': 1260759180},
    ])
    actual_dict = build_item_dict(data, 'movieId', 'title')
    expected_dict = {'31': 'Toy Story', '1029': 'Father of the Bride Part II (1995)'}
    TestCase().assertDictEqual(expected_dict, actual_dict)


def test_data_preprocessing_implicit_feedback(df):
    """
    Test data preprocessing for the case of implicit feedback data
    """
    processed_data = data_preprocessing_implicit_feedback(df, "userId", "movieId")
    assert len(processed_data) == df.shape[0] - 1
    assert processed_data.shape[1] == df.shape[1]
    assert 'Occur' in processed_data.columns
    expected = list([1, 2, 1, 1])
    actual = list(processed_data.Occur)
    assert actual == expected


class TestCreateTrainTestData(TestCase):
    """Test create_train_test_data method"""

    def setUp(self):
        """Setup interactions data."""
        self.interactions_pandas = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    def test_create_train_test_data(self):
        train_pd, test_pd, train_interactions, test_interactions = create_train_test_data(
            self.interactions_pandas,
            test_percentage=0.3,
            nice_print=True)
        self.assertIsInstance(train_pd, pd.DataFrame)
        self.assertIsInstance(test_pd, pd.DataFrame)
        self.assertIsInstance(train_interactions, coo_matrix)
        self.assertIsInstance(test_interactions, coo_matrix)
        # Note that Lightfm split method used ensure that the train and test to have the same size.
        # See: https://github.com/lyst/lightfm/blob/master/lightfm/lightfm.py
        self.assertEqual(train_pd.shape, (3, 2))
        self.assertEqual(test_pd.shape, (3, 2))
