import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint

df = pd.read_csv("trips_with_clusters.csv")
df["OriginTime"] = pd.to_datetime(df["OriginTime"])
df["date"] = df["OriginTime"].dt.date

if "AverageSpeed_kmh" not in df.columns:
    df["AverageSpeed_kmh"] = df["TripDistance_km"] / (df["TripDuration_min"] / 60)

# General distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df["TripDistance_km"], bins=30)
axes[0].set_title("Trip Distance Distribution")
axes[1].hist(df["TripDuration_min"], bins=30)
axes[1].set_title("Trip Duration Distribution")
axes[2].hist(df["AverageSpeed_kmh"], bins=30)
axes[2].set_title("Average Speed Distribution")
plt.tight_layout()
plt.show()

# Trips by departure hour
trip_counts = df.groupby("departure_hour").size()
plt.figure(figsize=(10, 5))
plt.bar(trip_counts.index, trip_counts.values)
plt.title("Number of Trips per Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Trips")
plt.tight_layout()
plt.show()

# Intra/inter-cluster movement
df["intra_cluster"] = df["origin_cluster"] == df["destination_cluster"]
total = df.groupby("origin_cluster").size().rename("TotalTrips")
intra = df[df["intra_cluster"]].groupby("origin_cluster").size().rename("IntraTrips")
cluster_stats = pd.concat([total, intra], axis=1).fillna(0)
cluster_stats["IntraPercent"] = 100 * cluster_stats["IntraTrips"] / cluster_stats["TotalTrips"]
cluster_stats["InterPercent"] = 100 - cluster_stats["IntraPercent"]

# Average maximum daily trip distance
max_daily = (df.groupby(["origin_cluster", "sensor", "date"])["TripDistance_km"]
               .max().groupby("origin_cluster").mean().rename("AvgMaxTripDistance_km"))
cluster_stats = cluster_stats.join(max_daily)

# Approximate spatial area of each origin cluster
areas = {}
for cluster_id, group in df.groupby("origin_cluster"):
    points = group[["Origin_X", "Origin_Y"]].drop_duplicates().values
    if len(points) < 3:
        areas[cluster_id] = 0.01
    else:
        areas[cluster_id] = MultiPoint(points).convex_hull.area * 12364

cluster_stats["Area_km2"] = pd.Series(areas)
cluster_stats["TripDensity"] = cluster_stats["TotalTrips"] / cluster_stats["Area_km2"]
cluster_stats["AvgSpeed_kmh"] = df.groupby("origin_cluster")["AverageSpeed_kmh"].mean()

summary = cluster_stats[["TripDensity", "AvgMaxTripDistance_km", "AvgSpeed_kmh",
                         "IntraPercent", "InterPercent"]].round(2)
print(summary)

plt.figure(figsize=(8, 6))
plt.scatter(summary["AvgMaxTripDistance_km"], summary["AvgSpeed_kmh"])
for cluster_id, row in summary.iterrows():
    plt.annotate(str(cluster_id), (row["AvgMaxTripDistance_km"], row["AvgSpeed_kmh"]))
plt.title("Average Maximum Trip Distance vs Average Speed")
plt.xlabel("Average Maximum Trip Distance (km)")
plt.ylabel("Average Speed (km/h)")
plt.grid(True)
plt.tight_layout()
plt.show()
