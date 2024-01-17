import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from tools import P

# Load data from https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive-concrete.csv
df = pd.read_csv("archive/concrete.csv")
X = df.copy()
y = X.pop("CompressiveStrength")

P(X.columns, 'purple')
P(X.head(), 'red')

X["FCRatio"] = X["FineAggregate"] / X["CoarseAggregate"]
X["AggCmtRatio"] = (X["CoarseAggregate"] + X["FineAggregate"]) / X["Cement"]
X["WtrCmtRatio"] = X["Water"] / X["Cement"]

model = RandomForestRegressor(criterion="absolute_error", random_state=0)
score = cross_val_score(
    model, X, y, cv=5, scoring="neg_mean_absolute_error"
)
score = -1 * score.mean()

P(f"MAE Score with Ratio Features: {score:.4}", 'green')
