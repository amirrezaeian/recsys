from unittest import TestCase
import pandas as pd
from typing import Dict
from lightfm import LightFM
from src.rec_sys import (fit_model, recommendations_items_for_user,
                         recommendations_users_for_item, embedding_matrix,
                         recommendations_items_for_item)


class TestRecommendation(TestCase):
    """Test recommendation for multiple cases."""
    @staticmethod
    def df() -> pd.DataFrame:
        """
        Build user-item interaction data for testing methods.
        """
        data = pd.DataFrame()
        data.index.name = "userId"
        data.columns.name = "movieId"

        data.loc['A', '1029'] = 1.0
        data.loc['A', '1129'] = 0.0
        data.loc['A', '1172'] = 0.0
        data.loc['A', '31'] = 1.0

        data.loc['B', '1029'] = 0.0
        data.loc['B', '1129'] = 1.0
        data.loc['B', '1172'] = 1.0
        data.loc['B', '31'] = 0.0
        return data

    @staticmethod
    def items_dict() -> Dict[str, str]:
        """Define item dictionary for testing."""
        items_dict = {'31': 'Toy Story',
                      '1029': 'Father of the Bride Part II (1995)',
                      '1172': 'Titanic',
                      '1129': 'Tomorrow'}
        return items_dict

    def test_fit_model(self) -> LightFM:
        """Test fit method of the model."""
        model = fit_model(self.df(), n_components=10, loss='bpr', k=3, epoch=20, n_jobs=1)
        assert model
        assert model.get_params()['no_components'] == 10
        assert model.get_params()['loss'] == 'bpr'
        assert model.get_params()['k'] == 3
        assert len(model.get_user_representations()) == 2
        return model

    def test_recommendations_items_for_user(self) -> None:
        """Test recommendation items for a user"""

        users_dict = {'A': 0, 'B': 1}

        recommendations_for_user_items = recommendations_items_for_user(self.test_fit_model(),
                                                                        self.df(),
                                                                        user_id='A',
                                                                        user_dict=users_dict,
                                                                        item_dict=self.items_dict(),
                                                                        threshold=1,
                                                                        num_recommendations=4)
        assert len(recommendations_for_user_items) == 2
        assert set(recommendations_for_user_items) == {'1172', '1129'}

    def test_recommendations_users_for_item(self) -> None:
        """Test recommendation similar users for an item."""

        recommendations_for_item_similar_users = recommendations_users_for_item(self.test_fit_model(),
                                                                                self.df(),
                                                                                item_id='31',
                                                                                number_of_user=2)
        assert len(recommendations_for_item_similar_users) == 2
        assert set(recommendations_for_item_similar_users) == {'A', 'B'}

    def test_user_not_exist(self) -> None:
        """Test error raised if user does not exist"""
        try:
            users_dict = {'A': 0, 'B': 1}
            items_dict = self.items_dict()
            _ = recommendations_items_for_user(self.test_fit_model(),
                                               self.df(),
                                               user_id='AA',
                                               user_dict=users_dict,
                                               item_dict=items_dict,
                                               threshold=1,
                                               num_recommendations=4)
        except ValueError as e:
            self.assertEqual(type(e), ValueError)
        else:
            TestCase.fail('ValueError not raised')

    def test_embedding_matrix(self) -> pd.DataFrame:
        """Test building embedding Matrix"""
        embedding_matrix_pandas = embedding_matrix(self.test_fit_model(), self.df())

        assert embedding_matrix_pandas.shape[0] == self.df().shape[1]
        assert embedding_matrix_pandas.shape[1] == self.df().shape[1]
        assert round(embedding_matrix_pandas.iloc[0, 0]) == 1
        assert round(embedding_matrix_pandas.iloc[1, 1]) == 1
        assert round(embedding_matrix_pandas.iloc[2, 2]) == 1
        assert round(embedding_matrix_pandas.iloc[3, 3]) == 1
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert round(embedding_matrix_pandas.iloc[i, j]) < 1
        return embedding_matrix_pandas

    def test_recommendations_items_for_item(self) -> None:
        """Test item-item recommendations"""

        embedding_matrix = self.test_embedding_matrix()
        item_id = '31'
        n_items = 3
        items_dict = self.items_dict()

        predicted_recommendation_items = recommendations_items_for_item(embedding_matrix,
                                                                        item_id,
                                                                        items_dict,
                                                                        n_items,
                                                                        False)
        expected_recommendation_items = ['1029', '1172', '1129']
        assert len(predicted_recommendation_items) == n_items
        TestCase().assertListEqual(predicted_recommendation_items, expected_recommendation_items)
