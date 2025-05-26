
"""
Penguin Species Clustering Project
----------------------------------

Objective:
----------
Help researchers identify groups of penguins in Antarctica using clustering methods on a dataset without known species labels.

Dataset:
--------
The dataset `penguins.csv` includes the following columns:

- culmen_length_mm: Culmen length (mm)
- culmen_depth_mm: Culmen depth (mm)
- flipper_length_mm: Flipper length (mm)
- body_mass_g: Body mass (g)
- sex: Penguin sex

Approach:
---------
1. Load and inspect the dataset.
2. Preprocess data: handle categorical features using dummy variables.
3. Standardize numerical features.
4. Use the Elbow method to determine optimal number of clusters.
5. Apply KMeans clustering.
6. Visualize clusters and summarize cluster statistics.

"""

# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1 - Loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")
print(penguins_df.head())
print(penguins_df.info())

# Step 2 - Perform preprocessing steps on the dataset to create dummy variables
# Convert categorical variables into dummy/indicator variables
penguins_df = pd.get_dummies(penguins_df, dtype='int')  # Ensures 0/1 output

# Step 3 - Standardizing the dataset
scaler = StandardScaler()
X = scaler.fit_transform(penguins_df)
penguins_preprocessed = pd.DataFrame(data=X, columns=penguins_df.columns)
print(penguins_preprocessed.head(10))

# Step 4 - Elbow method to determine optimal number of clusters
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(penguins_preprocessed)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Assuming optimal number of clusters from elbow method
n_clusters = 4

# Step 5 - Apply KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(penguins_preprocessed)
penguins_df['label'] = kmeans.labels_

# Visualize the clusters using culmen_length_mm
plt.scatter(penguins_df['label'], penguins_df['culmen_length_mm'], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Cluster')
plt.ylabel('Culmen Length (mm)')
plt.xticks(range(int(penguins_df['label'].min()), int(penguins_df['label'].max()) + 1))
plt.title(f'K-means Clustering (K={n_clusters})')
plt.show()

# Step 6 - Create final summarized DataFrame
numeric_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'label']
stat_penguins = penguins_df[numeric_columns].groupby('label').mean()
print(stat_penguins)
