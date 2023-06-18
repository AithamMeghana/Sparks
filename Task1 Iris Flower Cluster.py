#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


# In[2]:


# Load the iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)


# In[3]:


# Perform K-means clustering with different numbers of clusters
max_clusters = 10
wcss = []
for n_clusters in range(1, max_clusters+1):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(iris_df)
    wcss.append(kmeans.inertia_)


# In[4]:


# Plotting the within-cluster sum of squares (WCSS) to find the optimal number of clusters
plt.figure(figsize=(8, 6))
plt.plot(range(1, max_clusters+1), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[9]:


# Perform K-means clustering with the optimal number of clusters
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(iris_df)
cluster_labels = kmeans.labels_


# In[6]:


# Add cluster labels to the dataset
iris_df['Cluster'] = cluster_labels


# In[8]:


# Visualize the clusters using pairplots
sns.set(style="ticks")
sns.pairplot(iris_df, hue="Cluster")
plt.show()


# In[16]:


# Plot the clusters and centroids in a 2D scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(iris_df.iloc[:, 0], iris_df.iloc[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.show()


# In[ ]:




