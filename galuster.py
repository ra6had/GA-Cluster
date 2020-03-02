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


class Generation:

	def __init__(self, size, n_clusters, n_variables, env):
		"""
		Instantiate a population of chromosomes of either MeanChrom or VarChrom

		Parameters
		==========
		size: Integer.
			The number of distinct individual solutions in the population.

		ch_type: string.
			The type of chromosome to be instantiated.

		env: array-like.
			The environment with which the generation interacts

		"""

		self.size = size
		self.population = []
		self.sorted_scores = []
		self.n_clusters = n_clusters
		self.n_variables = n_variables
		self.scores = []
		self.env = env

		for chrom in range(self.size):
			chrom = MeanChrom(self.n_clusters, self.n_variables)
			self.population.append(chrom.chrom)
			



	def __str__(self):
		return str(self.population)


	"""Score using numpy"""
	def score(self):
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

		for chromosome in self.population:
			dist = KMeans(self.n_clusters, chromosome, 1).fit_transform(self.env)
			sd = []
			for i in dist:
				sd.append(min(i))
			scores.append(sum(sd))
		

		
		self.scores = np.array(scores)
		self.sorted_scores = np.argsort(self.scores)
		return self.scores



	def select(self, survival_rate=0.5):
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
			self.score()
#			self.sorted_scores = np.argsort(self.score())
			n = (len(self.sorted_scores))*survival_rate
			for i in range(int(n)):
				survivors.append(self.population[self.sorted_scores[i]])
		else:
			pass

		self.population = survivors
		self.size = len(survivors)
		return self.population




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
		
		for chrom in self.population:
			mutant_chrom = chrom.flatten() # flatten to simplify
			for i in range(len(mutant_chrom)):
				num = np.random.random()
				if num < mutation_rate: #apply mutation per gene
					mutant_chrom[i] = np.random.random()
				else:
					mutant_chrom[i] = mutant_chrom[i]
			mutant_chrom = mutant_chrom.reshape(self.n_clusters,
									   self.n_variables)
			mutant_pop.append(mutant_chrom)

		self.population = mutant_pop
		return mutant_pop




	def breed(self, cut_off=0.2, method='random'):#		generation = []
		seq = np.random.permutation(self.size) #Create random sequence
		pairs = seq.reshape(-1,2) #Match pairs as per random sequence
		
		if method != 'random' and method != 'spatial':
			raise ValueError('Please pass in a valid argument for the method n\
				   parameter. Accepted values are "random" & "spatial"')
		elif method == 'random':
			if cut_off >= 1:
				raise ValueError('cut_off rate argument must be less than 1')
				
			else:
				n = int(cut_off * self.n_clusters)
	
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
	
					self.population.append(child_one)
					self.population.append(child_two)

		
		elif method == 'spatial':
			for pair in pairs:
				x_chrom = self.population[pair[0]]
				y_chrom = self.population[pair[1]]
				distances = []
				for x_vector in x_chrom:
					for y_vector in y_chrom:
						distances.append(dist.euclidean(x_vector, y_vector))
				#xdist_mat = np.array(distances)
				xdist_mat = np.reshape(distances,(-1, self.n_clusters))
				ydist_mat = xdist_mat.transpose()
				child_one = []
				child_two = []
				for i in range(self.n_clusters):
					gene_one = (x_chrom[i] + y_chrom[np.argmin(xdist_mat[i])]) * 0.5
					child_one.append(gene_one)
					
					gene_two = (y_chrom[i] + x_chrom[np.argmin(ydist_mat[i])]) * 0.5
					child_two.append(gene_two)
					
				self.population.append(np.array(child_one))
				self.population.append(np.array(child_two))

		#self.population = np.array(self.population)
		return self.population