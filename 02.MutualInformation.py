import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from tools import P

plt.style.use("seaborn-v0_8-whitegrid")
df = pd.read_csv("archive/autos.csv")
X = df.copy()
y = X.pop("price")

P(f'This was the data types: \n{X.dtypes}', 'red')
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()
P(f'This is the data types: \n{X.dtypes}', 'purple')


discrete_features = (X.dtypes == int)
P(f'This is the data types: \n{discrete_features}', 'green')

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
P(mi_scores, 'blue')

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.savefig("mi_scores.png")

plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)

sns.relplot(x="curb_weight", y="price", data=df);
plt.savefig("curb_weight_price.png")

sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df);
plt.savefig("horsepower_price.png")
