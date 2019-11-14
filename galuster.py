import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise
import scipy.spatial.distance as dist



class MeanChrom:

	def __init__(self, n_clusters, n_variables):
		
		"""
		Instantiates a chromosome of cluster centroids to be used as initial seeds
		for K-means algorithm
	
		Arguments
		=========
		
		n_clusters: Intiger. Number of clusters
		
		n_variables: Intiger. Number of variables
	
		"""
		
		self.n_clusters = n_clusters
		self.n_variables = n_variables

		self.chrom = np.random.random((self.n_clusters, self.n_variables))



	def __str__(self):
		return "This is the meanest chromosome ever!"




class VarChrom:

	def __init__(self, n_variables, n_features):
		"""
		Instantiates a chromosome of variables, used in feature reduction
		
		Arguments
		=========
		
		n_variables: Integer. Total number of variables in the dataset
		
		n_features: Integer. Number of features to be used in classification.
			Must be less than n_variables.
	
		"""
		
		self.n_variables = n_variables
		self.n_features = n_features
		n_zeros = n_variables - n_features
		zeros = np.zeros(n_zeros)
		ones = np.ones(self.n_features)
		self.chrom = np.hstack((zeros, ones)).astype(int)
		np.random.shuffle(self.chrom)



	def __str__(self):
		return self.chrom


class Population:
	
	def __init__(self, size, ch_type='means', env=[], **kwargs):
		"""
		Instantiates a population of chromosomes of either MeanChrom or VarChrom
		
		Arguments
		==========
		size: intiger. The number of chromosomes in the population
		
		ch_type: string. the type of chromosome to be instantiated
		
		env: 2D array like object
		
		kwargs: the arguments for the selected chromosoe type
		
		"""		
				
		self.size = size
		self.ch_type = ch_type
		self.pop = []
		self.env = env

		for chrom in range(self.size):

			if self.ch_type != 'means' and self.ch_type != 'variables':
				raise ValueError('Please input a valid chromosome type n\
					 Supported chromosome types are "means" and "variables"')
				break
			elif self.ch_type == 'means':
				chrom = MeanChrom(kwargs['n_clusters'], kwargs['n_variables'])
				self.pop.append(chrom.chrom)
			elif self.ch_type == 'variables':
				chrom = VarChrom(kwargs['n_variables'], kwargs['n_features'])
				self.pop.append(chrom.chrom)
			else:
				raise ValueError('Make sure you insert the appropriate kwargs')



	def __str__(self):
		return self.pop



	def score(self, env=X):
		self.X = X
		scores = []
        
		if self.ch_type == 'means':
			n_clusters = len(self.pop[0])
			means = []
			for chromosome in self.pop:
				kmeans = KMeans(n_clusters, chromosome).fit(self.X)
				centers = kmeans.cluster_centers_
				means.append(centers)
				clusters = kmeans.predict(self.X)
				distances = []
				for i in range(len(clusters)):
					distance = dist.euclidean(centers[clusters[i]], self.X[i])
					distances.append(distance)
				scores.append(sum(distances))
		elif self.ch_type == 'variables':
			pass
		
		return scores
		
#	def score(self, X):
#		scores = pd.DataFrame(columns=['score'])
#		
#		if self.ch_type == 'means':
#			n_clusters = len(self.pop[0])
#			means = []
#			for i in range(len(self.pop)):
#				kmeans = KMeans(n_clusters, self.pop[i]).fit(X)
#				centers = kmeans.cluster_centers_
#				means.append(centers)
#				clusters = kmeans.predict(X)
#				distances = []
#				for i in range(len(clusters)):
#					distance = dist.euclidean(centers[clusters[i]], X[i])
#					distances.append(distance)
#				scores.loc[len(scores)] = sum(distances)
#		elif self.ch_type == 'variables':
#			pass
#		
		return scores
			
	
	def select(self, survival_rate=0.5, method='fittest'):
		
