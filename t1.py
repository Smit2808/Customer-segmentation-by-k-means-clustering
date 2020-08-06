import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


df = pd.read_csv('Mall_Customers.csv')

df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Annual Income (k$)'] = pd.to_numeric(df['Annual Income (k$)'], errors='coerce')
df['Spending Score (1-100)'] = pd.to_numeric(df['Spending Score (1-100)'], errors='coerce')
#print(df.describe())

def grp (row):
    if row['Gender'] == "Male":
        return 1
    elif row['Gender'] == "Female":
        return 0

df['grp'] = df.apply (lambda row: grp (row),axis=1)
 
cluster = df[['CustomerID', 'Age', 'grp', 'Annual Income (k$)', 'Spending Score (1-100)']]
print(cluster.describe())

# Let's scale the data first
scaler = StandardScaler()
df_scaled = scaler.fit_transform(cluster)

print(df_scaled.shape)
print()

print(df_scaled)

# applying elbow method to obtain optimal numbers of clusters
scores = []

for i in range(1,19):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df_scaled)
    scores.append(kmeans.inertia_)
    
plt.plot(scores, 'bx-')
plt.show()

 # By elbow method we got optimal cluster as 6.
kmeans = KMeans(6)
kmeans.fit(df_scaled)
labels = kmeans.labels_

print()
print(kmeans.cluster_centers_.shape)

print()
cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [cluster.columns])
print(cluster_centers)

# we got cluster centers by above 2 lines of code but in order to understand what these numbers mean, let's perform inverse transformation
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [cluster.columns])
print(cluster_centers)

#for labeling all data
print()
print(labels.shape)

print()
print(labels.max())

print()
print(labels.min())

print()
y_kmeans = kmeans.fit_predict(df_scaled)
print(y_kmeans)

print()
# concatenate the clusters labels to our original dataframe
df_cluster = pd.concat([df, pd.DataFrame({'cluster':labels})], axis = 1)
print(df_cluster.head())

print()
# Obtain the principal components
pca = PCA(n_components=2)
principal_comp = pca.fit_transform(df_scaled)
print(principal_comp)

print()
pca_df = pd.DataFrame(data = principal_comp, columns =['pca1','pca2'])
print(pca_df.head())

print()
pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)
print(pca_df.head())

plt.figure(figsize=(10,10))
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','pink','black','gray'])
plt.show()