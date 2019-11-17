import galuster
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans  as km
import scipy.spatial.distance as dist


df = pd.read_csv('sample.csv').set_index('OA')
X = df.values

#kmeans = km(n_clusters=8).fit_predict(X)



pop = galuster.Population(10, 'means', n_clusters=9, n_variables=10)

#score = []
#if pop.ch_type == 'means':
#    means = []
#    n_clusters = len(pop.pop[0])
#    for chromosome in pop.pop:
#        kmeans = km(n_clusters, chromosome).fit(X)
#        centers = kmeans.cluster_centers_
#        means.append(centers)
#        clusters = kmeans.predict(X)
#        distances = []
#        for i in range(len(clusters)):
#            distance = dist.euclidean(centers[clusters[i]], X[i])
#            distances.append(distance)
#        score.append(sum(distances))
#            

population = pop.pop
scores = pop.score(X)

survivors = pop.select(X)


