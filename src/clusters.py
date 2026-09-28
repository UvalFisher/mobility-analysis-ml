import pandas as pd
import folium
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("detected_trips_filtered.csv")
all_points = pd.concat([
    df[["Origin_X", "Origin_Y"]].rename(columns={"Origin_X": "X", "Origin_Y": "Y"}),
    df[["Destination_X", "Destination_Y"]].rename(columns={"Destination_X": "X", "Destination_Y": "Y"})
], ignore_index=True)

wcss = []
for k_candidate in range(1, 16):
    model = KMeans(n_clusters=k_candidate, n_init="auto", random_state=0)
    model.fit(all_points[["X", "Y"]])
    wcss.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 16), wcss, marker="o")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS (Inertia)")
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_plot.png")
plt.show()

k = 4
kmeans = KMeans(n_clusters=k, random_state=0).fit(all_points[["X", "Y"]])
df["origin_cluster"] = kmeans.predict(df[["Origin_X", "Origin_Y"]].rename(columns={"Origin_X": "X", "Origin_Y": "Y"}))
df["destination_cluster"] = kmeans.predict(df[["Destination_X", "Destination_Y"]].rename(columns={"Destination_X": "X", "Destination_Y": "Y"}))
df.to_csv("trips_with_clusters.csv", index=False)

colors = plt.cm.get_cmap("tab20", k).colors
colors = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b, _ in colors]
center_lat = pd.concat([df["Origin_Y"], df["Destination_Y"]]).mean()
center_lon = pd.concat([df["Origin_X"], df["Destination_X"]]).mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

for _, row in df.iterrows():
    folium.CircleMarker([row["Origin_Y"], row["Origin_X"]], radius=3,
                        color=colors[row["origin_cluster"]], fill=True, fill_opacity=0.6).add_to(m)
    folium.CircleMarker([row["Destination_Y"], row["Destination_X"]], radius=3,
                        color=colors[row["destination_cluster"]], fill=True, fill_opacity=0.6).add_to(m)

m.save("trip_clusters_map.html")
print("Saved: trips_with_clusters.csv and trip_clusters_map.html")
