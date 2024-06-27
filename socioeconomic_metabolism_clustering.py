#%%
# Library
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import geopandas as gpd
# %%

# Open SEM_Variables.xlsx
sem_file_location = "C:/Users/Aldis/Documents/Master Data Science/GitHub/Urban_Metabolism_24/Data/Week_2_Data/SEM_data/SEM_Variables.xlsx"
sem_data = pd.read_excel(sem_file_location)
print(sem_data.head())
#%%

# Create a data frame od sem_data
sem_df = pd.DataFrame(sem_data)
sem_df.head()
#%%

# keep with the columns: Year, country, GDP in 2015 EUR/inhab, Direct extraction and Imports
sem_df = sem_df[['Year', 'Country', 'GDP in 2015 EUR/inhab', 'DE_IN (ton/cap)', 'IMP_IN (ton/cap)']]
sem_df.head()
#%%

# Create a column with the name of DMI = IMP_IN (ton/cap) + DE_IN (ton/cap), and remove the DE_IN (ton/cap) and IMP_IN (ton/cap) column
sem_df['DMI'] = sem_df['IMP_IN (ton/cap)'] + sem_df['DE_IN (ton/cap)']
sem_df.drop(['DE_IN (ton/cap)', 'IMP_IN (ton/cap)'], axis=1, inplace=True)
sem_df.head()
#%%

# Group by Country and do the average of GDP in 2015 EUR/inhab and DMI of all years. Remove the year column
sem_df_mean = sem_df.groupby('Country').mean().reset_index()
sem_df_mean = sem_df_mean.drop(['Year'], axis=1)
sem_df_mean.head()
#%%


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Select the features for clustering
X = sem_df_mean[['GDP in 2015 EUR/inhab', 'DMI']]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use the elbow method 
wcss = []
max_clusters = 10
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the WCSS to visualize the elbow method
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, max_clusters + 1))
plt.grid(True)
plt.show()
#%%

# Select the features for clustering
X = sem_df_mean[['GDP in 2015 EUR/inhab', 'DMI']]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform k-means clustering 
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Add the cluster labels to the original DataFrame
sem_df_mean['Cluster'] = kmeans.labels_

# Print the DataFrame with the cluster labels
print(sem_df_mean)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
plt.title('K-means Clustering')
plt.xlabel('GDP in 2015 EUR/inhab (scaled)')
plt.ylabel('DMI (scaled)')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
#%%

# Data frame for country with cluster = 0
cluster_0_df = sem_df_mean[sem_df_mean['Cluster'] == 0]
print(cluster_0_df)
#%%
# Data frame for country with cluster = 1
cluster_1_df = sem_df_mean[sem_df_mean['Cluster'] == 1]
print(cluster_1_df)
#%%
# Data frame for country with cluster = 2
cluster_2_df = sem_df_mean[sem_df_mean['Cluster'] == 2]
print(cluster_2_df)
#%%



# Hierarchical Cluster
# Select the features for clustering
X = sem_df_mean[['GDP in 2015 EUR/inhab', 'DMI']]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute the linkage matrix using Complete linkage
linked = linkage(X_scaled, method='complete')

# Number of cluster
num_clusters = 3
clusters = fcluster(linked, num_clusters, criterion='maxclust')

# Add the cluster labels to the original DataFrame
sem_df_mean['Cluster'] = clusters

# List of countries in each cluster
clustered_countries = {}
for cluster_num in range(1, num_clusters + 1):
    countries_in_cluster = sem_df_mean.loc[sem_df_mean['Cluster'] == cluster_num, 'Country'].tolist()
    clustered_countries[f'Cluster {cluster_num}'] = countries_in_cluster

# Print the list of countries in each cluster
for cluster, countries in clustered_countries.items():
    print(f"Cluster {cluster}: {', '.join(countries)}")

# Visualize the dendrogram (optional)
plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', labels=sem_df_mean['Country'].tolist(), distance_sort='descending')
plt.title('Hierarchical Clustering Dendrogram (Complete linkage)')
plt.xlabel('Countries')
plt.ylabel('Distance')
plt.show()
#%%


# GDP and DMI graph

# Define the list of countries in the cluster
Cluster_same_1 = ['Austria', 'Belgium', 'Denmark', 'Netherlands', 'Sweden']
# Filter the countries of the list Cluster_same_1 in the data frame = sem_df
Cluster_same_1_df = sem_df[sem_df['Country'].isin(Cluster_same_1)]
Cluster_same_1_df.head()

# Plot GDP in 2015 EUR/inhab vs Year for each country in the same plot
fig, ax = plt.subplots(figsize=(10, 6))

# Loop through each country in the cluster
for country in Cluster_same_1:
    # Filter the data for the current country
    country_data = Cluster_same_1_df[Cluster_same_1_df['Country'] == country]

    # Plot the data
    ax.plot(country_data['Year'], country_data['GDP in 2015 EUR/inhab'], label=country)

# Set the title and labels
ax.set_title('GDP in 2015 EUR/inhab vs Year for Cluster 2 Countries')
ax.set_xlabel('Year')
ax.set_ylabel('GDP in 2015 EUR/inhab')

# Add a legend and grid
ax.legend()
ax.grid(True)

# Show the plot
plt.show()
#%%

# Plot DMI vs Year for each country in the same plot
# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Loop through each country in the cluster
for country in Cluster_same_1:
    # Filter the data for the current country
    country_data = Cluster_same_1_df[Cluster_same_1_df['Country'] == country]

    # Plot the data
    ax.plot(country_data['Year'], country_data['DMI'], label=country)

# Set the title and labels
ax.set_title('DMI vs Year for Cluster 2 Countries')
ax.set_xlabel('Year')
ax.set_ylabel('DMI')

# Add a legend and grid
ax.legend()
ax.grid(True)

# Show the plot
plt.show()
#%%



# Map of Europ
shapefile_path = "C:/Users/Aldis/Documents/Master Data Science/GitHub/Urban_Metabolism_24/Data/Week_2_Data/country_shapefile/Europe_merged.shp"  

# File
shp = gpd.read_file(shapefile_path)
shp.rename(columns={"COUNTRY": "Country"}, inplace=True)

# Clusters
cluster_1 = ['Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Estonia', 'Greece', 'Hungary', 'Italy', 'Latvia', 'Lithuania', 'Malta', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Turkey']
cluster_2 = ['Austria', 'Belgium', 'Denmark', 'Netherlands', 'Sweden']
cluster_3 = ['Norway']

# Clusters to countrys
shp['Custom_Cluster'] = shp['Country'].apply(
    lambda x: 1 if x in cluster_1 else (2 if x in cluster_2 else (3 if x in cluster_3 else 0))
)

# Plot the clusters 
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Colors
shp.plot(ax=ax, column='Custom_Cluster', categorical=True, legend=True, cmap='tab10')

# Leyend configuration
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=['Cluster 1', 'Cluster 2', 'Cluster 3'], loc='lower left', bbox_to_anchor=(0.8, 0.5))

# Names
for idx, row in shp.iterrows():
    plt.annotate(text=row['Country'], xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
                 xytext=(3, 3), textcoords="offset points")

plt.title('Clustering Visualization by Country')

# Plot
plt.show()
# %%
