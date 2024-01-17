import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

accidents = pd.read_csv("archive/accidents.csv")
autos = pd.read_csv("archive/autos.csv")
concrete = pd.read_csv("archive/concrete.csv")
customer = pd.read_csv("archive/customer.csv")


autos["stroke_ratio"] = autos.stroke / autos.bore
P(autos[["stroke", "bore", "stroke_ratio"]].head(), 'blue')

autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders
)

P(autos[["displacement"]].head(), 'red')

accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)
P(accidents[["LogWindSpeed"]].head(), 'yellow')

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(accidents.WindSpeed, shade=True, ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, shade=True, ax=axs[1]);
plt.savefig("wind_speed.png")


roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
    "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
    "TrafficCalming", "TrafficSignal"]
accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)
P(accidents[roadway_features + ["RoadwayFeatures"]].head(10), 'green')


components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
               "Superplasticizer", "CoarseAggregate", "FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1)

P(concrete[components + ["Components"]].head(10), 'red')


customer[["Type", "Level"]] = (customer["Policy"].str.split(" ", expand=True))
P(customer[["Policy", "Type", "Level"]].head(10), 'yellow')

autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
P(autos[["make", "body_style", "make_and_style"]].head(), 'blue')


customer["AverageIncome"] = (customer.groupby("State")["Income"].transform("mean"))

P(customer[["State", "Income", "AverageIncome"]].head(10), 'purple')

customer["StateFreq"] = (customer.groupby("State")["State"].transform("count") / customer.State.count())

P(customer[["State", "StateFreq"]].head(10), 'cyan')

df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)


df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

df_valid = df_valid.merge(df_train[["Coverage", "AverageClaim"]].drop_duplicates(), on="Coverage", how="left",)
P(df_valid[["Coverage", "AverageClaim"]].head(10), 'white')
