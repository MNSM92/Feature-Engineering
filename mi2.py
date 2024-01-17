import numpy as np
import pandas as pd

# Load data
accidents = pd.read_csv("./archive/accidents.csv")
autos = pd.read_csv("./archive/autos.csv")
concrete = pd.read_csv("./archive/concrete.csv")
customer = pd.read_csv("./archive/customer.csv")

# Calculate the ratio of stroke to bore
autos["stroke_ratio"] = autos.stroke / autos.bore

# Display the first five rows of the new column
print(autos[["stroke", "bore", "stroke_ratio"]].head())

# Calculate the number of missing values in each column
autos["displacement"] = ( np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders )

# If the feature has 0.0 values, use np.log1p (log(1+x)) instead of np.log
accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)

# Plot a comparison
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(accidents.WindSpeed, shade=True, ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, shade=True, ax=axs[1]);

# Create a feature for the total number of road features
roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
    "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
    "TrafficCalming", "TrafficSignal"]

# Create a feature for total number of road features
accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)

# Display the head of the accidents dataset
print(accidents[roadway_features + ["RoadwayFeatures"]].head(10))


# Create dummy variables for `RoadwayFeatures`
components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
               "Superplasticizer", "CoarseAggregate", "FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1)

# Display first 10 rows of the dataset
concrete[components + ["Components"]].head(10)

# Create dummy variables for `Type` and `Level`
customer[["Type", "Level"]] = (  # Create two new features
    customer["Policy"]           # from the Policy feature
    .str                         # through the string accessor
    .split(" ", expand=True)     # by splitting on " "
                                 # and expanding the result into separate columns
)

# Display first 10 rows of `customer`
customer[["Policy", "Type", "Level"]].head(10)

# Create a feature representing the engine volume
autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
autos[["make", "body_style", "make_and_style"]].head()

# Groupby and mean
customer["AverageIncome"] = (
    customer.groupby("State")  # for each state
    ["Income"]                 # select the income
    .transform("mean")         # and compute its mean
)

customer[["State", "Income", "AverageIncome"]].head(10)

# Groupby and count
customer["StateFreq"] = (
    customer.groupby("State")
    ["State"]
    .transform("count")
    / customer.State.count()
)

customer[["State", "StateFreq"]].head(10)

# Create splits
df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)

# Create the average claim amount by coverage type, on the training set
df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

# Merge the values into the validation set
df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
    on="Coverage",
    how="left",
)

df_valid[["Coverage", "AverageClaim"]].head(10)
