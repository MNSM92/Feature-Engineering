import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from tools import P

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

df = pd.read_csv("archive/housing.csv")
X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]
P(X.head(), 'purple')

# Create cluster feature
kmeans = KMeans(n_clusters=6, n_init='auto', n)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")
P(X.head(), 'blue')
sns.relplot(
    x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
);
plt.savefig("kmeans.png")

# Add cluster column to dataframe
X["MedHouseVal"] = df["MedHouseVal"]
sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6);
plt.savefig("kmeans2.png")
