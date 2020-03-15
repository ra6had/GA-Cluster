import galuster
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans  as km
import matplotlib.pyplot as plt
from datetime import datetime



#Define Variables
gens = 20
gen_size = 52
n_clusters = 2
n_seed = 10000
no_kmeans = 1
cluster_list = []
sum_of_dist = []
comp_duration = []


#Import and prepare OA data for clustering
X = []
with open('LOAC_Input_Data.csv') as file:
	df = pd.read_csv(file).set_index('OA')
	X = df.values


#Instantiate initial generation
init_pop = galuster.Generation(gen_size, n_clusters=n_clusters, n_variables=60, env=X)

#Iterate GA operations over number of generations
pop = init_pop
fittest = []
top_scores = []
#survivors = []
ga_start = datetime.now()

for i in range(gens):
	print('Generation no: ' + str(i +1))
	pop.select()
	#survivors.append(pop.population)
	top_scores.append((min(pop.score())))
	fittest.append(pop.population[pop.sorted_scores[0]])
	pop.mutate(0.001)
	pop.breed(method='hybrid')



#Cluster the data using the fittest seed
init_pop.population = fittest
init_pop.sorted_scores = np.argsort(init_pop.score())
fit_rank = init_pop.sorted_scores
ga_means = km(n_clusters, fittest[fit_rank[0]],1).fit(X)

ga_end = datetime.now()
comp_duration.append(ga_end - ga_start)

cluster_list.append(ga_means)
sum_of_dist.append(galuster.sum_distances(ga_means, X))

#Plot GA cluster membership
plt.figure(0)
galuster.lolipop_plot(ga_means, X)

for i in range(no_kmeans):
	print('Starting kmeans algorithm no: ' + str(i + 1))
	start = datetime.now() #Record starting time
	kmeans = km(n_clusters, n_init=n_seed).fit(X) #compute kmeans
	end = datetime.now() #Record ending time
	cluster_list.append(kmeans) #Append to cluster list
	sum_of_dist.append(galuster.sum_distances(kmeans, X))
	comp_duration.append(end - start)

	#Plot cluster membership
	plt.figure(i+1)
	galuster.lolipop_plot(kmeans, X)


