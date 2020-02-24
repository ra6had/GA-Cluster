import galuster
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans  as km
#import scipy.spatial.distance as dist


X = []
with open('LOAC_Input_Data.csv') as file:
	df = pd.read_csv(file).set_index('OA')
	X = df.values
	
#df = pd.read_csv('LOAC_Input_Data.csv').set_index('OA')
#X = df.values

#kmeans = km(n_clusters=8).fit_predict(X)

gens = 50
n_clusters = 9

init_pop = galuster.Generation(100, n_clusters=n_clusters, n_variables=60, env=X)


pop = init_pop
fittest = []
top_scores = []
for i in range(gens):
	survivors = pop.select()
	top_scores.append((min(pop.score())))
	fittest.append(pop.population[pop.sorted_scores[0]])
	pop.mutate(0.001)
	pop.breed(0.02)

#kmeans = km(9, fittest[-1]).fit(X)

init_pop.population = fittest
#scores = init_pop.score(X)
init_pop.sorted_scores = np.argsort(init_pop.score())
fit_rank = init_pop.sorted_scores
kmeans = km(n_clusters, fittest[fit_rank[0]]).fit(X)
distances = kmeans.transform(X)
sd = 0
for i in distances:
	sd = sd + min(i)
	




#population = init_pop.population
#scores = init_pop.score(X)
#
#survivors = init_pop.select(X)
#init_pop.population = survivors
#mutants = init_pop.mutate(0.5)
