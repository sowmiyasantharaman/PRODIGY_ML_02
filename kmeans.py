import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('Mall_Customers.csv')

categorical_columns = ['Gender']
data_encoded = pd.get_dummies(data, columns=categorical_columns)
numeric_columns = data_encoded.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
data_encoded[numeric_columns] = scaler.fit_transform(data_encoded[numeric_columns])

k = 3
kmeans = KMeans(n_clusters=k, random_state=0)

kmeans.fit(data_encoded)

clusters = kmeans.labels_

data['Cluster'] = clusters
data.to_csv('clustered_customers.csv', index=False)
