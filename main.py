import galuster
import pandas as pd
#import numpy as np
from sklearn.cluster import KMeans  as km
#import scipy.spatial.distance as dist


df = pd.read_csv('LOAC_Input_Data.csv').set_index('OA')
X = df.values

#kmeans = km(n_clusters=8).fit_predict(X)



init_pop = galuster.Generation(100, 'means', n_clusters=7, n_variables=60)

gens = 10

pop = init_pop
fittest = []
for i in range(gens):
	survivors = pop.select(X)
	print("scores of this generation are: /n" + str(pop.score(X)))
	fittest.append(pop.population[pop.sorted_scores[0]])
	pop.mutate(0.01)
	pop.breed()

kmeans = km(7, fittest[-1]).fit(X)








#population = init_pop.population
#scores = init_pop.score(X)
#
#survivors = init_pop.select(X)
#init_pop.population = survivors
#mutants = init_pop.mutate(0.5)
