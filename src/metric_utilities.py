"""Build metric utilities."""
import logging
import pandas as pd
from typing import Dict, Any
from scipy.sparse import coo_matrix
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def model_metrics(
        model: LightFM,
        train_interactions: coo_matrix,
        test_interactions: coo_matrix,
        k: int = 10,
        no_epochs: int = 100,
        nice_plot: bool = True,
        **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    Get model performance metrics and plot them.

    :param model: fitted LightFM model
    :param train_interactions: (scipy sparse COO matrix),  train interactions set
    :param test_interactions: (scipy sparse COO matrix), test interaction set
    :param k: number of recommendations, optional
    :param no_epochs: Number of epochs to run, optional
    :param  nice_plot: if True, the metric results will be plotted.
    :param **kwargs: other keyword arguments to be passed

    Output: Pandas Dateframe containing the metrics.
    """

    precision_train, precision_test = [0] * no_epochs, [0] * no_epochs
    recall_train, recall_test = [0] * no_epochs, [0] * no_epochs
    auc_train, auc_test = [0] * no_epochs, [0] * no_epochs

    for epoch in range(no_epochs):
        model.fit_partial(interactions=train_interactions, **kwargs)
        precision_train[epoch] = precision_at_k(model, train_interactions, k=k, **kwargs).mean()
        precision_test[epoch] = precision_at_k(model, test_interactions, k=k, **kwargs).mean()

        recall_train[epoch] = recall_at_k(model, train_interactions, k=k, **kwargs).mean()
        recall_test[epoch] = recall_at_k(model, test_interactions, k=k, **kwargs).mean()

        auc_train[epoch] = auc_score(model, train_interactions, **kwargs).mean()
        auc_test[epoch] = auc_score(model, test_interactions, **kwargs).mean()

    data_epoch = pd.DataFrame(
        zip(precision_train, precision_test, recall_train, recall_test, auc_train, auc_test),
        columns=[
            "precision@k (train)",
            "precision@k (test)",
            "recall@k (train)",
            "recall@k (test)",
            "AUC (train)",
            "AUC (test)"])

    auc_train_avg = data_epoch["AUC (train)"].mean()
    auc_test_avg = data_epoch["AUC (test)"].mean()
    precision_train_avg = data_epoch["precision@k (train)"].mean()
    precision_test_avg = data_epoch["precision@k (test)"].mean()
    recall_train_avg = data_epoch["recall@k (train)"].mean()
    recall_test_avg = data_epoch["recall@k (test)"].mean()

    df_result = pd.DataFrame(columns=['Evaluation Metric', 'Train', 'Test'])
    df_result = df_result.append(pd.DataFrame([['Average AUC', round(auc_train_avg, 2),
                                                round(auc_test_avg, 2)],
                                               ['Average Precision@{}'.format(k),
                                                round(precision_train_avg, 2),
                                                round(precision_test_avg, 2)],
                                               ['Average recall@{}'.format(k),
                                                round(recall_train_avg, 2),
                                                round(recall_test_avg, 2)]],
                                              columns=df_result.columns))

    if nice_plot:
        df_auc = data_epoch[['AUC (train)', 'AUC (test)']]
        if not df_auc.isna().any().any():
            plot_1 = df_auc.plot(style=['o', 'rx'], title="AUC score")
            plot_1.legend(['Train', 'Test'])
            plot_1.set_xlabel("epoch")
            plot_1.set_ylabel("value")

        df_precision = data_epoch[['precision@k (train)', 'precision@k (test)']]
        if not df_precision.isna().any().any():
            plot_2 = data_epoch[['precision@k (train)', 'precision@k (test)']].plot(
                style=['o', 'rx'], title="precision@{}".format(k))
            plot_2.legend(['Train', 'Test'])
            plot_2.set_xlabel("epoch")
            plot_2.set_ylabel("value")

        df_recall = data_epoch[['recall@k (train)', 'recall@k (test)']]
        if not df_recall.isna().any().any():
            plot_3 = df_recall.plot(style=['o', 'rx'], title="recall@{}".format(k))
            plot_3.legend(['Train', 'Test'])
            plot_3.set_xlabel("epoch")
            plot_3.set_ylabel("value")

    return df_result
