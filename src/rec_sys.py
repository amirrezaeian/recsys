"""Build model for recsys."""
import logging
import pandas as pd
import numpy as np
from scipy import sparse
from lightfm import LightFM
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List


SEED = 42

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def fit_model(interactions: pd.DataFrame, n_components: int = 30,
              loss: str = 'warp', k: int = 15,
              epoch: int = 30, n_jobs: int = 1) -> LightFM:
    """
    Build matrix-factorization LightFM model

    :param interactions: dataset create by create_interaction_matrix
    :param n_components:  the dimensionality of the feature latent embeddings.
    :param loss: loss function, one of (‘logistic’, ‘bpr’, ‘warp’, ‘warp-kos’).
    :param k: the k-th positive example will be selected from the n positive examples sampled for every user.
    :param epoch:  number of epochs to run
    :param n_jobs: Number of parallel computation threads to use.
    Should not be higher than the number of physical cores.

    Output: trained model
    """
    assert k > 0
    assert epoch > 0
    assert k > 0
    assert n_components > 0
    assert n_jobs > 0
    assert loss in ("logistic", "warp", "bpr", "warp-kos")

    x = sparse.csr_matrix(interactions.values)
    model = LightFM(no_components=n_components, loss=loss, k=k,
                    random_state=np.random.RandomState(SEED))
    model.fit(x, epochs=epoch, num_threads=n_jobs)
    return model


def recommendations_items_for_user(model: LightFM, interactions: pd.DataFrame, user_id: str,
                                   user_dict: Dict[str, int], item_dict: Dict[str, str],
                                   threshold: float = 1,
                                   num_recommendations: int = 10,
                                   nice_print: bool = True) -> List[str]:
    """
    Produce new recommendations (not already clicked or liked) for a user_id.

    :param model: Trained matrix factorization model
    :param interactions: dataset used for training the model
    :param user_id: user ID for which we need to generate recommendation
    (this id is an actual id from outset)
    :param user_dict: Dictionary containing interaction_index as key and
    an integer mapped user_id as value
    :param item_dict: Dictionary containing item_id as key and item_name as value
    :param threshold: value above which the rating is favorable
    :param num_recommendations: Number of output recommendation request
    :param nice_print: if true the results will be printed out (this is for notebook usage).

    Output: list of recommended ids (sorted by score)
    """
    assert num_recommendations > 0

    n_users, n_items = interactions.shape
    if user_id in user_dict:
        user_x = user_dict[user_id]
    else:
        raise ValueError('user id is not found. Make sure user_dict is updated')

    scores = pd.Series(model.predict(user_x, np.arange(n_items)))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))

    known_items = list(pd.Series(interactions.loc[user_id, :][interactions.loc[user_id, :] >=
                                                              threshold].index
                                 ).sort_values(ascending=False))

    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:num_recommendations]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))

    if nice_print:
        print("Known Likes:")
        for i, value in enumerate(known_items, start=1):
            print(str(i) + '- ' + value)

        print("\n Recommended new items:")
        for i, value in enumerate(scores, start=1):
            print(str(i) + '- ' + value)
    return return_score_list


def recommendations_users_for_item(model: LightFM, interactions: pd.DataFrame,
                                   item_id: str, number_of_user: int = 2) -> List[str]:
    """
    Recommend a list of top N (number_of_user) users for a given item.

    :param model:  Trained matrix factorization model
    :param interactions:  dataset used for training the model
    :param item_id: item ID for which we need to generate recommended users
    :param number_of_user: Number of users needed as an output

    Output: List of recommended users
    """
    assert number_of_user > 0

    n_users, n_items = interactions.shape
    x = np.array(interactions.columns)
    if item_id in x:
        x_index = x.searchsorted(item_id)
    else:
        raise ValueError('item id is not found. Make sure user-item interaction data is updated')
    scores = pd.Series(model.predict(np.arange(n_users), np.repeat(x_index, n_users)))
    user_list: List[str] = list(interactions.index[scores.sort_values(ascending=False).head(number_of_user).index])
    return user_list


def embedding_matrix(model: LightFM, interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build item-item distance embedding matrix

    :param model: Trained matrix factorization model. Call fit_model method to built this.
    :param interactions: dataset used for training the model. Call build_interaction_matrix method to build this.

    Output:
    Pandas dataframe containing cosine distance matrix between items obtained from embedding vectors
    """
    embedding_vectors = model.item_embeddings
    data_embeddings = sparse.csr_matrix(embedding_vectors)
    items_cosine_similarities = cosine_similarity(data_embeddings)
    items_cosine_similarities_pandas = pd.DataFrame(items_cosine_similarities)
    items_cosine_similarities_pandas.columns = interactions.columns
    items_cosine_similarities_pandas.index = interactions.columns
    return items_cosine_similarities_pandas


def recommendations_items_for_item(built_embedding_matrix: pd.DataFrame,
                                   item_id: str,
                                   item_dict: Dict[str, str],
                                   n_items: int = 10,
                                   nice_print: bool = True) -> List[str]:
    """
    Build item-item recommendation: Recommend N (n_items) items similar to item_id

    :param built_embedding_matrix: Pandas dataframe containing cosine distance matrix between items
    call embedding_matrix method to build it.
    :param item_id: item id that we need to generate recommended items
    :param item_dict: Dictionary containing item_id as key and item_name as value
    :param n_items: Number of items needed for recommendation
    :param nice_print: if True, it prints the results. This is only useful for notebook.

    Output: a list of recommended items
    """
    all_recommended_items = pd.Series(built_embedding_matrix.loc[item_id, :].sort_values(ascending=False))
    n_items_recommended = all_recommended_items.head(n_items + 1).index[1:n_items + 1]
    n_items_recommended_list = list(n_items_recommended)

    if nice_print:
        print(f'Similar items to "{item_dict[item_id]}":')
        for counter, value in enumerate(n_items_recommended_list, start=1):
            print(str(counter) + '- ' + item_dict[value])
    return n_items_recommended_list
