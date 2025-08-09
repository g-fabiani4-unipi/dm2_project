from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.utils import _safe_indexing
from sklearn.utils._response import _get_response_values_binary
import matplotlib.pyplot as plt
import numpy as np

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