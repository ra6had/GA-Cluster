import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise
import scipy.spatial.distance as dist



class MeanChrom:

	def __init__(self, n_clusters, n_variables):
		
		"""
		Instantiates a chromosome of cluster centroids to be used as initial
		seeds for K-means algorithm
	
		Parameters
		==========
		
		n_clusters: Integer.
			Number of clusters.
		
		n_variables: Integer.
			Number of variables.
	
		"""
		
		self.n_clusters = n_clusters
		self.n_variables = n_variables
		
		#Create random centroids in the variable space
		self.chrom = np.random.random((self.n_clusters, self.n_variables))



	def __str__(self):
		return "This is the meanest chromosome ever!"




class VarChrom:

	def __init__(self, n_variables, n_features):
		"""
		Instantiate a chromosome of binary genes used for feature reduction.
		Each gene in the chromosome represents to a variable. A gene value of
		1 means the corresponding variable is included in the classification.
		
		Parameters
		=========
		
		n_variables: Integer.
			Total number of variables in the dataset
		
		n_features: Integer. 
			Number of features to be used in classification. Must be less
			than n_variables.
	
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


class Generation:
	
	def __init__(self, population, ch_type='means'):
						
		self.size = len(population)
		self.population = population

	def __str__(self):
		return "This is a generation of " + str(self.size) + " chromosomes"	



	def generation(size, n_variables, ch_type='means', **kwargs):
		
		for chrom in range(size):

			if self.ch_type != 'means' and self.ch_type != 'variables':
				raise ValueError('Please input a valid chromosome type n\
					 Supported chromosome types are "means" and "variables"')
				break
			elif self.ch_type == 'means':
				chrom = MeanChrom(kwargs['n_clusters'], n_variables)
				self.pop.append(chrom.chrom)
			elif self.ch_type == 'variables':
				chrom = VarChrom(n_variables, kwargs['n_features'])
				self.pop.append(chrom.chrom)
			else:
				raise ValueError('Make sure you insert the appropriate kwargs')		



	"""Score using numpy"""
	def score(self, env):		
		"""
		Score the members of a population in relation to a given environment.
		
		Parameters
		=========
		env: array_like.
			2D array containg the attributes of the objects to be classified
			where each row represents one object.
		
		Returns
		=======
		scores: NumPy Array.
			A 1D NumPy array of length equal to population size. Each value
			is the sum of 
			
		"""
		
		scores = []
        
		if self.ch_type == 'means':
			
			#Get number of clusters from chromosome in population. 
			n_clusters = len(self.pop[0])
			#means = []
			
			#Run & evaluate K-means using every chromosome as initial seed
			for chromosome in self.pop:
				kmeans = KMeans(n_clusters, chromosome).fit(env)
				centers = kmeans.cluster_centers_
				#means.append(centers)
				clusters = kmeans.predict(env) #Cluster all objects
				distances = [] 
				
				#Compute distance between each object and its cluster's center
				for i in range(len(clusters)):
					#
					distance = dist.euclidean(centers[clusters[i]], env[i])
					distances.append(distance)
				
				scores.append(sum(distances)) #Compute sum of distances
		
		elif self.ch_type == 'variables':
			pass
		
		return np.array(scores)

	"""Score using pandas"""		
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
#		return scores
			
	
	
	
	def select(self, env, survival_rate=0.5):
		"""
		Select a proportion of the population to breed and pass on their
		chromosomes to the next generation.
		
		Parameters
		==========
		env: array_like.
			2D array containg the attributes of the objects to be classified
			where each row represents one object.
		survival_rate: float.
			A rate representing the percentage of the population to select for
			breeding.

		Returns
		=======
		survivors: list.
			A list of NumPy arrays, each NumPy array is a selected chromosome
		"""
		
		if survival_rate >= 1:
			raise ValueError('survival_rate argument must be less than 1')
		elif survival_rate < 1:
			survivors = []
		
			#Score and sort the population by score
			sorted_scores = np.argsort(self.score(env))
			n = (len(sorted_scores))*survival_rate
			for i in range(int(n)):
				survivors.append(self.pop[sorted_scores[i]])
		else:
			pass
		
		return survivors
	
	
	
	def mutate(self, mutation_rate=0.01):
		pass
	
	

	def breed(survivors, cut_off=0.5, method='random'):
		
		if cut_off >= 1:
			raise ValueError('cut_off rate argument must be less than 1')
		elif method != 'random' and method != 'diverse':
			raise ValueError('Please pass in a valid argument for the method n\
				   parameter. Accepted values are "random" & "diverse"')
		elif method == 'random':
			
			
		pass