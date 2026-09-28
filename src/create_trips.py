import pandas as pd
from math import radians, sin, cos, sqrt, atan2


def haversine(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return radius_km * c


df = pd.read_csv("cellular_data_enriched.csv")
df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
df = df.sort_values(["sensor", "TimeStamp"]).reset_index(drop=True)

trips = []
for sensor_id, group in df.groupby("sensor"):
    group = group.reset_index(drop=True)
    i, n = 1, len(group)

    while i < n:
        row = group.loc[i]
        if row["AerialDistance_km"] > 2 and 0 < row["TravelDuration_min"] < 30 and row["AerialSpeed_kmh"] > 5:
            origin = group.loc[i - 1]
            start_index = i - 1
            max_distance = row["AerialDistance_km"]
            j = i + 1

            while j < n:
                next_row = group.loc[j]
                dist = haversine(origin["Y"], origin["X"], next_row["Y"], next_row["X"])
                if (row["AerialDistance_km"] < 2 or next_row["TravelDuration_min"] > 30 or
                    next_row["AerialSpeed_kmh"] < 5 or dist < max_distance or
                    (dist == max_distance and next_row["TravelDuration_min"] > 0)):
                    break
                if dist > max_distance or (dist == max_distance and next_row["TravelDuration_min"] == 0):
                    max_distance = dist
                j += 1

            if j - 1 > start_index:
                destination = group.loc[j - 1]
                trip_duration = (destination["TimeStamp"] - origin["TimeStamp"]).total_seconds() / 60
                trip_distance = haversine(origin["Y"], origin["X"], destination["Y"], destination["X"])

                if trip_duration >= 20:
                    avg_speed = trip_distance / (trip_duration / 60)
                    if 5 <= avg_speed <= 70:
                        trips.append({
                            "sensor": sensor_id, "OriginTime": origin["TimeStamp"],
                            "Origin_X": origin["X"], "Origin_Y": origin["Y"],
                            "DestinationTime": destination["TimeStamp"],
                            "Destination_X": destination["X"], "Destination_Y": destination["Y"],
                            "TripDuration_min": round(trip_duration, 1),
                            "TripDistance_km": round(trip_distance, 1),
                            "AverageSpeed_kmh": round(avg_speed, 1)
                        })
                i = j
            else:
                i += 1
        else:
            i += 1

trips_df = pd.DataFrame(trips)
print(f"Total trips: {len(trips_df)}")
trips_df["departure_hour"] = pd.to_datetime(trips_df["OriginTime"]).dt.hour

def assign_hour_bin(hour):
    if hour < 6: return 0
    if hour < 9: return 6
    if hour < 12: return 9
    if hour < 15: return 12
    if hour < 18: return 15
    if hour < 21: return 18
    return 21

trips_df["hour_bin"] = trips_df["departure_hour"].apply(assign_hour_bin)
trips_df.to_csv("detected_trips_filtered.csv", index=False)
print("Saved: detected_trips_filtered.csv")
