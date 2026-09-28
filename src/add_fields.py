import pandas as pd
import numpy as np


def haversine_np(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance in kilometers."""
    radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius_km * c


df = pd.read_csv("cellular_data.csv")
df = df[(df["X"] != 0) & (df["Y"] != 0)].copy()
df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
df = df.sort_values(["sensor", "TimeStamp"]).reset_index(drop=True)

aerial_distances, durations, speeds = [], [], []

for _, group in df.groupby("sensor"):
    group = group.sort_values("TimeStamp")
    lat, lon = group["Y"].values, group["X"].values
    time = group["TimeStamp"].values.astype("datetime64[s]").astype("int")

    distance = haversine_np(lat[:-1], lon[:-1], lat[1:], lon[1:])
    duration_min = (time[1:] - time[:-1]) / 60
    speed_kmh = distance / (duration_min / 60)

    aerial_distances.extend(np.insert(distance, 0, np.nan))
    durations.extend(np.insert(duration_min, 0, np.nan))
    speeds.extend(np.insert(speed_kmh, 0, np.nan))

df["AerialDistance_km"] = aerial_distances
df["TravelDuration_min"] = durations
df["AerialSpeed_kmh"] = speeds
df.to_csv("cellular_data_enriched.csv", index=False)
print("Saved: cellular_data_enriched.csv")
