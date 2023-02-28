"""Util methods for model training."""
import logging
import pandas as pd
import numpy as np
from scipy import sparse
from lightfm import cross_validation
from typing import Dict, Tuple

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

SEED = 40


def build_interaction_matrix(df: pd.DataFrame,
                             user_col: str,
                             item_col: str,
                             rating_col: str,
                             norm: bool = False,
                             threshold: float = 1.0,
                             nice_printing: bool = True) -> pd.DataFrame:
    """
    Build an interaction matrix dataframe from transactional type interactions.

    :param df: contains user-item interactions.
    :param user_col: column name for user's identifier.
    :param item_col: column name for item's identifier.
    :param rating_col: column name containing user feedback on interaction with a given item.
    :param norm: optional, True if a normalization of ratings is needed.
    :param threshold: value above which the rating is favorable. It is used if norm = True.
    :param nice_printing: If True it prints some information.

    Output: Pandas dataframe with user-item interactions ready to be fed in a recommendation algorithm.

    Note: before calling this method we need to validate the all required columns are available.
    If the required columns are not available, then errors should raise and this method should not be called.
    """
    interactions_matrix = df.groupby([user_col, item_col])[rating_col].sum().unstack().reset_index()
    interactions_matrix = interactions_matrix.fillna(0.0).set_index(user_col)

    if norm:
        interactions_matrix = interactions_matrix.applymap(lambda x: 1.0 if x >= threshold else 0.0)
    if nice_printing:
        print(f"Number of unique user in data: {interactions_matrix.shape[0]}")
        print(f"Number of unique product Id in data: {interactions_matrix.shape[1]}")

    return interactions_matrix


def build_user_dict(df_interactions: pd.DataFrame) -> Dict[str, int]:
    """
    Build a user dictionary based on their index and number in interaction dataset.

    :param df_interactions: dataset created by build_interaction_matrix method.

    Output: user_dict - Dictionary containing actual user_id in interaction_index as key
    and the position of the user in the interaction matrix as value. The value of this dict is internal id
    which models like LightFM consume for the predict method. 
    """
    user_id = list(df_interactions.index)
    user_dict = {user: counter for counter, user in enumerate(user_id)}
    return user_dict


def build_item_dict(df: pd.DataFrame, id_col: str, name_col: str) -> Dict[str, str]:
    """
    Build an item dictionary mapping between item_id and item name.


   :param df: Pandas dataframe with item information
   :param id_col: Column name containing unique identifier for an item
   :param name_col: Column name in the dataframe which contains item names.

    Output: Dictionary containing item_id as key and item_name as value
    """
    item_dict = df.set_index(id_col)[name_col].to_dict()
    return item_dict


def data_preprocessing_implicit_feedback(df: pd.DataFrame,
                                         user_col: str,
                                         item_col: str) -> pd.DataFrame:
    """
    Build data-frame for implicit feedback clicks.
    Add a new feature "Occur" which counts number of occurrence of item_col for a
    given user_col.

    :param df: contains user-item interactions.
    :param user_col: column name for user's identifier.
    :param item_col: column name for item's identifier.

    Output: Pandas dataframe with user-item interactions with a new feature "Occur".

    """
    df = df.copy()
    if 'timestamp' in df.columns:
        logger.info(f'timestamp feature exist in data, it will be dropped.')
        df = df.drop('timestamp', axis=1)
    df['Occur'] = df.groupby(item_col)[user_col].transform('size')
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def create_train_test_data(interactions_pandas: pd.DataFrame,
                           test_percentage: float,
                           nice_print: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                             sparse.coo_matrix,  sparse.coo_matrix]:
    """
    Create train and test data sets from the Interactions data.

    LightLM expects the train and test sets to have same dimension.
    Therefore, the conventional train test split will not work.

    :param interactions_pandas: Interactions dataframe.
    Call build_interaction_matrix method to build this.
    :param test_percentage: percentage of test data
    :param nice_print: if True it prints some information
    """
    interactions_csr = sparse.csr_matrix(interactions_pandas.values)

    train_interactions, test_interactions = cross_validation.random_train_test_split(
        interactions_csr,
        test_percentage=test_percentage,
        random_state=np.random.RandomState(SEED))

    train_pd = pd.DataFrame(train_interactions.todense())
    test_pd = pd.DataFrame(test_interactions.todense())
    train_pd.columns = test_pd.columns = interactions_pandas.columns
    train_pd.index = test_pd.index = interactions_pandas.index

    if nice_print:
        print(f"Shape of train interaction data: {train_interactions.shape}")
        print(f"Shape of test interaction data: {test_interactions.shape}")

    return train_pd, test_pd, train_interactions, test_interactions
