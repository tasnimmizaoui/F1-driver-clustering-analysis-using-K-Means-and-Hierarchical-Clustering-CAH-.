# F1-driver-clustering-analysis-using-K-Means-and-Hierarchical-Clustering-CAH-.

### 📁 `f1_driver_clustering_analysis.md`

# 🏎️ F1 Driver Clustering Analysis (1950–2024)

This project analyzes the performance of F1 drivers over seasons using clustering techniques (K-Means and Hierarchical Clustering). The analysis is based on driver points, wins, and final standings per season.

---

## 📊 1. Data Loading & Preprocessing

```python
import pandas as pd
import numpy as np

# Load CSV files
drivers = pd.read_csv('drivers.csv')
races = pd.read_csv('races.csv')
driver_standings = pd.read_csv('driver_standings.csv')
results = pd.read_csv('results.csv')
constructors = pd.read_csv('constructors.csv')
````

### 🛠️ Join & Clean

```python
# Merge to get year and driver info
standings = driver_standings.merge(races[['raceId', 'year']], on='raceId')
standings = standings.merge(drivers[['driverId', 'driverRef', 'surname']], on='driverId')

# Keep final race only per season per driver
final_standings = standings.sort_values('raceId', ascending=False).drop_duplicates(['driverId', 'year'])

# Clean columns
driver_season_df = final_standings[['driverId', 'driverRef', 'surname', 'year', 'points', 'wins', 'position']].copy()
driver_season_df['points'] = driver_season_df['points'].astype(float)
driver_season_df['wins'] = driver_season_df['wins'].astype(int)
driver_season_df['position'] = pd.to_numeric(driver_season_df['position'], errors='coerce')
driver_season_df = driver_season_df.dropna()

# Add binary target
driver_season_df['top3'] = driver_season_df['position'] <= 3
```

---

## 🔢 2. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

features = driver_season_df[['points', 'wins', 'position']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
```

---

## 🎯 3. K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

inertia = []
silhouette = []

K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X_scaled, kmeans.labels_))
```

### 📈 Elbow & Silhouette

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(K_range, inertia, marker='o'); ax[0].set_title('Inertie')
ax[1].plot(K_range, silhouette, marker='o', color='green'); ax[1].set_title('Silhouette')
plt.show()
```

### ✅ Final Clustering (K=4)

```python
kmeans_final = KMeans(n_clusters=4, random_state=42)
driver_season_df['cluster'] = kmeans_final.fit_predict(X_scaled)
```

---

## 📉 4. PCA for Visualization

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=driver_season_df['cluster'], cmap='Set1')
plt.title("K-Means (K=4) with PCA"); plt.grid(True); plt.show()
```

---

## 📊 5. Cluster Analysis

```python
# Silhouette Score
score = silhouette_score(X_scaled, driver_season_df['cluster'])

# Cluster Means
cluster_stats = driver_season_df.groupby('cluster')[['points', 'wins', 'position']].mean()

# Heatmap
import seaborn as sns
sns.heatmap(cluster_stats.T, annot=True, cmap='YlGnBu')
plt.title("Performance by Cluster")
plt.show()
```

---

## 🏅 6. Naming Clusters

```python
cluster_labels = {
    2: "Elite Champions",
    0: "Strong Contenders",
    3: "Midfield Performers",
    1: "Backmarkers"
}

driver_season_df['cluster_label'] = driver_season_df['cluster'].map(cluster_labels)
```

### 📍 Final Scatter Plot with Labels

```python
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=driver_season_df['cluster_label'], palette='Set1')
plt.title("F1 Driver Performance Clusters")
plt.grid(True)
plt.show()
```

---

## 🧩 7. Hierarchical Clustering (CAH)

```python
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='lastp', p=20)
plt.axhline(y=15, color='r', linestyle='--')
plt.title("Dendrogram - CAH")
plt.show()
```

### 📌 Distance Jumps

```python
last_merges = Z[-20:, 2]
distance_jumps = np.diff(last_merges)

plt.plot(range(1, len(distance_jumps)+1), distance_jumps, 'o-')
plt.axvline(x=4, color='r', linestyle='--')
plt.title('Fusion Distance Jumps')
plt.show()
```

### 📍 CAH Clustering Result

```python
from sklearn.cluster import AgglomerativeClustering

cah = AgglomerativeClustering(n_clusters=4, linkage='ward')
driver_season_df['cluster_CAH'] = cah.fit_predict(X_scaled)

# Pairplot
sns.pairplot(driver_season_df, vars=['points', 'wins', 'position'], hue='cluster_CAH')
plt.show()
```

---

## 📌 Summary

* **Data:** Driver stats per season (points, wins, final position).
* **Methods:** K-Means and Hierarchical Clustering (Ward).
* **Best K:** 4 (based on silhouette & dendrogram).
* **Clusters Identified:**

  * Elite Champions
  * Strong Contenders
  * Midfield Performers
  * Backmarkers

---

> 📁 *This project gives insight into grouping driver-seasons into performance tiers using unsupervised learning.*

