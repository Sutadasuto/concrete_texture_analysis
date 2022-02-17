import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV


def get_best_feats(X, y, clf, folds, min_features_to_select, plot=False, step=1, sample_weights=None):

    # The "accuracy" scoring shows the proportion of correct classifications
    rfecv = RFECV(
        estimator=clf,
        step=step,
        cv=StratifiedKFold(folds, shuffle=True, random_state=0),
        scoring="f1_macro",
        min_features_to_select=min_features_to_select,
    )
    rfecv.fit(X, y, sample_weights)

    print("Optimal number of features : %d" % rfecv.n_features_)
    best_features = rfecv.support_

    if plot:
        # Plot number of features VS. cross-validation scores
        plt.title(str(type(clf)).split("'")[1].split(".")[-1])
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (%s)" % rfecv.scoring)
        n_feats_list = list(reversed([i for i in range(X.shape[1], -X.shape[1], -step) if i > min_features_to_select - step]))
        # Sklearn ensures that minimum n. of features is 2. Then, any value below 2 in the RFE is set to 2
        for idx, n in enumerate(n_feats_list):
            if n < 2:
                n_feats_list[idx] = 2
        plt.plot(
            # range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
            n_feats_list,
            rfecv.grid_scores_,
        )
        plt.xlim([min_features_to_select, X.shape[1]])
        plt.ylim([0, 1])

    return best_features
