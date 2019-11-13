import galuster
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans  as km


df = pd.read_csv('sample.csv').set_index('OA')
X = df.values

kmeans = km(n_clusters=8).fit_predict(X)



pop = galuster.Population(10, 'means', n_clusters=5, n_variables=10)

score = []
if pop.ch_type == 'means':
    n_clusters = len(pop.pop[0])
    for chromosome in pop.pop:
        kmeans = km(n_clusters, chromosome).fit(X)
        centers = kmeans.cluster_centers_
        clusters = kmeans.predict(X)
        for element in clusters:
            dist = np.linalg.norm(centers[element], X[index(element)])
            