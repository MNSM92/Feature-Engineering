import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

plt.style.use("seaborn-v0_8-whitegrid")

df = pd.read_csv("archive/autos.csv")

X = df.copy()
y = X.pop("price")

# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = X.dtypes == int

from sklearn.feature_selection import mutual_info_regression
import pandas as pd

def make_mi_scores(X, y, discrete_features):
    """
    Calculate mutual information scores for each feature in X with respect to target variable y.

    Parameters:
    X (DataFrame): The input features.
    y (Series): The target variable.
    discrete_features (array-like): List of indices or boolean mask indicating which features are
                                    discrete. If empty, assume all features are continuous.

    Returns:
    mi_scores (Series): The mutual information scores for each feature.
    """

    # Calculate mutual information scores
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)

    # Create a Series with the scores, using feature names as index
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)

    # Sort the scores in descending order
    mi_scores = mi_scores.sort_values(ascending=False)

    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
print(mi_scores[::3])
