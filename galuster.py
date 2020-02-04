import numpy as np
#import pandas as pd
from sklearn.cluster import KMeans
#from sklearn.metrics import pairwise
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
		==========

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

	def __init__(self, size, ch_type='means', **kwargs):
		"""
		Instantiate a population of chromosomes of either MeanChrom or VarChrom

		Parameters
		==========
		size: Integer.
			The number of distinct individual solutions in the population.

		ch_type: string.
			The type of chromosome to be instantiated.

		kwargs: the Parameters for the selected chromosome type. If the
			chromosome represents cluster means then two addition

		"""

		self.size = size
		self.ch_type = ch_type
		self.population = []
		self.sorted_scores = []

		for chrom in range(self.size):

			if self.ch_type != 'means' and self.ch_type != 'variables':
				raise ValueError('Please input a valid chromosome type n\
					 Supported chromosome types are "means" and "variables"')
				break
			elif self.ch_type == 'means':
				chrom = MeanChrom(kwargs['n_clusters'], kwargs['n_variables'])
				self.population.append(chrom.chrom)
			elif self.ch_type == 'variables':
				chrom = VarChrom(kwargs['n_variables'], kwargs['n_features'])
				self.population.append(chrom.chrom)
			else:
				raise ValueError('Make sure you insert the appropriate kwargs')



	def __str__(self):
		return str(self.population)


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
			n_clusters = len(self.population[0])
			#means = []

			#Run & evaluate K-means using every chromosome as initial seed
			for chromosome in self.population:
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
#			n_clusters = len(self.population[0])
#			means = []
#			for i in range(len(self.population)):
#				kmeans = KMeans(n_clusters, self.population[i]).fit(X)
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
			self.sorted_scores = np.argsort(self.score(env))
			n = (len(self.sorted_scores))*survival_rate
			for i in range(int(n)):
				survivors.append(self.population[self.sorted_scores[i]])
		else:
			pass

		return survivors




	def mutate(self, mutation_rate=0.01):
		"""
		Randomly mutate genes in individuals of the self Generation at a user-
		defined probability.

		Parameters
		==========
		mutation_rate: the probability for any given gene to be mutated

		MODIFIES THE Generation.population() ATTRIBUTE!!
		"""

		mutant_pop = []
		clusters = len(self.population[0]) #Number of clusters
		variables = len(self.population[0][0]) #Number of variables


		if self.ch_type == "means":
			for chromosome in self.population:
				mutant_chrom = chromosome.flatten() #Flatten to simplify
				for i in range(len(mutant_chrom)):
					num = np.random.random()
					if num < mutation_rate: #applied per gene
						mutant_chrom[i] = np.random.random()
					else:
						mutant_chrom[i] = mutant_chrom[i]
				mutant_chrom = mutant_chrom.reshape(clusters, variables)
				mutant_pop.append(mutant_chrom)



		elif self.ch_type == "variables":
			pass

		self.population = mutant_pop
		return mutant_pop




	def breed(self, cut_off=0.2, method='random'):

		n_chrom = len(self.population)
		clusters = len(self.population[0])
		variables = len(self.population[0][0])
		generation = []

		if cut_off >= 1:
			raise ValueError('cut_off rate argument must be less than 1')
		elif method != 'random' and method != 'diverse':
			raise ValueError('Please pass in a valid argument for the method n\
				   parameter. Accepted values are "random" & "diverse"')
		elif method == 'random':


			if self.ch_type == "means":
				seq = np.random.permutation(n_chrom) #Create random sequence
				pairs = seq.reshape(-1,2) #Match pairs as per random sequence
				n = int(cut_off * clusters)

				#From every pair breed two offsprings
				for pair in pairs:
					x_chrom = [self.population[pair[0]][:n],
							self.population[pair[1]][:n]]

					y_chrom = [self.population[pair[1]][n:],
							self.population[pair[0]][n:]]

					child_one = np.concatenate((x_chrom[0],
								 y_chrom[0]), axis=0)

					child_two = np.concatenate((x_chrom[1],
								 y_chrom[1]), axis=0)

					generation.append(child_one)
					generation.append(child_two)




			elif self.ch_type == "variables":
				seq = np.random.permutation(variables)
				pairs = seq.reshape(-1,2)
				for pair in pairs:
					pass
				pass
		elif method == 'diverse':
			dist_matrix = []
			for chrom in self.population:
				for center in chrom:
					pass

			pass
		return generation