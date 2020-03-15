import galuster
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans  as km
import matplotlib.pyplot as plt
from datetime import datetime
import copy



#Define Variables
gens = 5
gen_size = 1000
n_clusters = 19
cluster_list = []
sum_of_dist = []
comp_duration = []
methods = ['random', 'spatial', 'mixed_spatial', 'hybrid']


#Import and prepare OA data for clustering
X = []
with open('LOAC_Input_Data.csv') as file:
	df = pd.read_csv(file).set_index('OA')
	X = df.values


#Instantiate initial generation
init_pop = galuster.Generation(gen_size, n_clusters=n_clusters, n_variables=60, env=X)

Fittest = []
Top_scores = []
#Survivors = []
Distances = []

for method in methods:
	fittest = []
	top_scores = []
	survivors = []
	print('Computing using: ' + method + ' breed method')
	start = datetime.now()
	GENERATION = copy.deepcopy(init_pop)
	for i in range(gens):
		print('Generation no: ' + str(i +1))
		GENERATION.select()
		survivors.append(GENERATION.population)
		top_scores.append((min(GENERATION.score())))
		fittest.append(GENERATION.population[GENERATION.sorted_scores[0]])
		GENERATION.mutate(0.001)
		GENERATION.breed(method=method)
	GENERATION.population = fittest
	GENERATION.sorted_scores = np.argsort(GENERATION.score())
	fit_rank = GENERATION.sorted_scores
	alpha = fittest[fit_rank[0]]
	ga_means = km(n_clusters, alpha, 1).fit(X)
	cluster_list.append(ga_means)
	end = datetime.now()
	comp_duration.append(end - start)
	Fittest.append(alpha)
	#Survivors.append(survivors)

for i in range(len(cluster_list)):
	Distances.append(galuster.sum_distances(cluster_list[i], X))
	plt.figure(i)
	galuster.lolipop_plot(cluster_list[i], X)



#Iterate GA operations over number of generations

#ga_start = datetime.now()
#
#for i in range(gens):
#	print('Generation no: ' + str(i +1))
#	GENERATION.select()
#	survivors.append(GENERATION.population)
#	top_scores.append((min(GENERATION.score())))
#	fittest.append(GENERATION.population[GENERATION.sorted_scores[0]])
#	GENERATION.mutate(0.001)
#	GENERATION.breed(method='mixed_spatial')
#
#
##Cluster the data using the fittest seed
#GENERATION.population = fittest
#GENERATION.sorted_scores = np.argsort(GENERATION.score())
#fit_rank = GENERATION.sorted_scores
#ga_means = km(n_clusters, fittest[fit_rank[0]],1).fit(X)
#
#ga_end = datetime.now()
#comp_duration.append(ga_end - ga_start)
#
#cluster_list.append(ga_means)
#sum_of_dist.append(galuster.sum_distances(ga_means, X))
#
##Plot GA cluster membership
#plt.figure(0)
#galuster.lolipop_plot(ga_means, X)

#for i in range(no_kmeans):
#	print('Starting kmeans algorithm no: ' + str(i + 1))
#	start = datetime.now() #Record starting time
#	kmeans = km(n_clusters, n_init=n_seed).fit(X) #compute kmeans
#	end = datetime.now() #Record ending time
#	cluster_list.append(kmeans) #Append to cluster list
#	sum_of_dist.append(galuster.sum_distances(kmeans, X))
#	comp_duration.append(end - start)
#
#	#Plot cluster membership
#	plt.figure(i+1)
#	galuster.lolipop_plot(kmeans, X)


