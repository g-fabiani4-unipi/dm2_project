from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.utils import _safe_indexing
from sklearn.utils._response import _get_response_values_binary
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def draw_pr_curve_from_cv_results(cv_results, X, y, name:str=None, color:str=None, ax=None):
    """
    Draw a Cross Validated PR Curve.


    Parameters:
    -------------
    cv_results: scores returned by a call to cross_validate

    name: name of an estimator

    Largely taken from: https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
    """
    title = "PR Curve"
    name = name if name else "Mean PR"
    default_line_kwargs = {"drawstyle": "steps-post"}


    if not ax:
        _, ax = plt.subplots(figsize=(6, 6))

    y_real = []
    y_proba = []

    i = 0
    for estimator, indices, ap in zip(
        cv_results['estimator'],
        cv_results['indices']['test'],
        cv_results['test_average_precision']
        ):
        y_true = _safe_indexing(y, indices)
        y_score, _ = _get_response_values_binary(
            estimator,
            _safe_indexing(X, indices),
            response_method='auto'
        )
        # Compute ROC curve and area the curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)

        # Plotting each individual PR Curve
        if color:
            ax.plot(recall, precision, lw=1, alpha=0.3, color=color, **default_line_kwargs)
        else:
            ax.plot(recall, precision, lw=1, alpha=0.3,
                    label='PR fold %d (AP = %0.2f)' % (i, ap),
                    **default_line_kwargs)

        y_real.append(y_true)
        y_proba.append(y_score)

        i += 1

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    ax.plot(recall, precision, color=color if color else 'b',
             label= name + r' (AP = %0.2f $\pm$ %0.2f)' % (np.mean(cv_results['test_average_precision']), np.std(cv_results['test_average_precision'])),
             lw=2, alpha=.8, **default_line_kwargs)

    ax.set(
        xlim=(-0.05, 1.05),
        ylim=(-0.05, 1.05),
        xlabel='Recall',
        ylabel='Precision',
        title=title
    )

    ax.legend(bbox_to_anchor=(1, 1))



def plot_silhouette(X, cluster_labels, ax=None):
    """
    Draw silhouette scores for all clustered samples

    Parameters:
    -----------
    X: dataset on which the clustering is made or square distance matrix

    cluster_labels: labels returned by a clustering estimator

    ax: (optional) axes on which to plot
    """

    if not ax:
        _, ax = plt.subplots()
    n_clusters = len(np.unique(cluster_labels))
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(1, n_clusters + 1):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = f'C{i}'
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="k", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks


def print_cv_results(cv_results, ndigits=2):
    for key, value in cv_results.items():
        if key.startswith('test_'):
            print(f'{key.removeprefix('test_')}: {round(np.mean(value), ndigits)} ({round(np.std(value), ndigits)})')


def cv_results_to_long(cv_results):
    cv_results = pd.DataFrame(cv_results)
    id_vars = [col for col in cv_results.columns if not col.startswith('split')]
    value_vars = [col for col in cv_results.columns if col.startswith('split')]
    long = cv_results.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='split',
        value_name='test_score'
    )
    return long
