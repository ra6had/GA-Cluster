import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import seaborn as sns



def breed_random_(population, n_clusters, cut_off, pairs):

	if cut_off >= 1:
		raise ValueError('cut_off rate argument must be less than 1')
	else:
		n = int(cut_off * n_clusters)

			#From every pair breed two offsprings
		for pair in pairs:
			x_chrom = [population[pair[0]][:n],
					population[pair[1]][:n]]

			y_chrom = [population[pair[1]][n:],
					population[pair[0]][n:]]

			child_one = np.concatenate((x_chrom[0],
						 y_chrom[0]), axis=0)
			population.append(child_one)
			
			
			child_two = np.concatenate((x_chrom[1],
						 y_chrom[1]), axis=0)

		
			population.append(child_two)

	return population


def breed_spatial_(population, n_clusters, pairs):

	for pair in pairs:
		x_chrom = population[pair[0]]
		y_chrom = population[pair[1]]
		distances = []
		for x_vector in x_chrom:
			for y_vector in y_chrom:
				distances.append(dist.euclidean(x_vector, y_vector))
		#xdist_mat = np.array(distances)
		xdist_mat = np.reshape(distances,(-1, n_clusters))
		ydist_mat = xdist_mat.transpose()
		child_one = []
		child_two = []
		for i in range(n_clusters):
			gene_one = (x_chrom[i] + y_chrom[np.argmin(xdist_mat[i])]) * 0.5
			child_one.append(gene_one)
		
			gene_two = (y_chrom[i] + x_chrom[np.argmin(ydist_mat[i])]) * 0.5
			child_two.append(gene_two)

			population.append(np.array(child_one))
		
			population.append(np.array(child_two))

	return population


def breed_mixed_spatial_(population, n_clusters, pairs):

	for pair in pairs:
		x_chrom = population[pair[0]]
		y_chrom = population[pair[1]]
		distances = []
		for x_vector in x_chrom:
			for y_vector in y_chrom:
				distances.append(dist.euclidean(x_vector, y_vector))
		#xdist_mat = np.array(distances)
		xdist_mat = np.reshape(distances,(-1, n_clusters))
		ydist_mat = xdist_mat.transpose()
		child_one = []
		child_two = []
		for i in range(n_clusters):
			gene_one = (x_chrom[i] + y_chrom[np.argmin(xdist_mat[i])]) * 0.5
			child_one.append(gene_one)
			
			gene_two = (y_chrom[i] + x_chrom[np.argmax(ydist_mat[i])]) * 0.5
			child_two.append(gene_two)

		population.append(np.array(child_one))
		population.append(np.array(child_two))

	return population


def breed_hybrid_(population, n_clusters, pairs):
	split = int(len(pairs/2))
	pairs_list = np.split(pairs, [split])
	first_half = breed_random_(population, n_clusters, 0.2, pairs_list[0])
	second_half = breed_spatial_(population, n_clusters, pairs_list[1])
	
	population = np.concatenate((population, first_half, second_half), axis=0)
	
	return population
		

def member_counts(kmeans, X):
	
	c_range = list(range(len(kmeans.cluster_centers_)))
	c_assignments = kmeans.predict(X)
	c_counts = []

	#Sum up cluster memberships
	for cluster in c_range:
		count = sum(1 if x==cluster else 0 for x in c_assignments)
		c_counts.append(count)

	#Link clusters with membership counts
	df = pd.DataFrame({'clusters':c_range, 'members':c_counts})
	df = df.sort_values(by='members')
	
	return df, c_range


def lolipop_plot(kmeans, X):
	"""
	Plots a lolipop plot showing the cluster membership for each cluster in
	the kmeans object

	Parameters:
		kmeans: an sklearn.cluster.KMeans object
		X: 2D array-like object containing the dataset to be clustered

	Returns:
		lolipop figure
	"""
	df, c_range = member_counts(kmeans, X)
#	#Prepare plot data
#	c_range = list(range(len(kmeans.cluster_centers_)))
#	c_assignments = kmeans.predict(X)
#	c_counts = []
#
#	#Sum up cluster memberships
#	for cluster in c_range:
#		count = sum(1 if x==cluster else 0 for x in c_assignments)
#		c_counts.append(count)
#
#	#Link clusters with membership counts
#	df = pd.DataFrame({'clusters':c_range, 'members':c_counts})
#	df = df.sort_values(by='members')

	#Create plot
	plt.hlines(y=c_range, xmin=0, xmax=df['members'], color='skyblue')
	plt.plot(df['members'], c_range, 'o')

	#Label plot
	plt.yticks(c_range, df['clusters'])
	plt.title('Cluster Membership Counts', loc='left')
	plt.xlabel('Member Counts')
	plt.ylabel('Clusters')

	#plt.show()



def sum_distances(kmeans, X):
	"""
	Compute the sum of distances between each object and its cluster center

	Parameters:
		kmeans: an sklearn.cluster.KMeans object
		X: 2D array-like object containing the dataset to be clustered

	Returns:
		sum_dist: float- sum of distances from cluster centers
	"""

	distances = kmeans.transform(X)
	sum_dist = 0
	for i in distances:
		sum_dist = sum_dist + min(i)
	return sum_dist




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
			distances = KMeans(self.n_clusters, chromosome, 1).fit_transform(self.env)
			sd = []
			for i in distances:
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

			#Compute number of surviving individuals
			n = (len(self.sorted_scores))*survival_rate
			for i in range(int(n)):
				survivors.append(self.population[self.sorted_scores[i]])
		else:
			pass

		#Update the Generations population
		self.population = survivors

		#Update the Generation size
		self.size = len(survivors)
		return self.population




	def mutate(self, mutation_rate=0.001):
		"""
		Randomly mutate genes in individuals of the self Generation at a user-
		defined probability.

		Parameters
		==========
		mutation_rate: the probability for any given gene to be mutated

		MODIFIES THE Generation.population() ATTRIBUTE!!
		"""

		#Instantiate new population container
		mutant_pop = []


		for chrom in self.population:
			mutant_chrom = chrom.flatten() # flatten to simplify
			for i in range(len(mutant_chrom)):
				num = np.random.random()
				if num < mutation_rate: #apply mutation rate per gene
					mutant_chrom[i] = np.random.random()
				else:
					mutant_chrom[i] = mutant_chrom[i]
			mutant_chrom = mutant_chrom.reshape(self.n_clusters,
									   self.n_variables)
			mutant_pop.append(mutant_chrom)

		#Update the Generation's population
		self.population = mutant_pop
		return mutant_pop







	def breed(self, method='random', cut_off=0.2)	:
	
		m = len(self.population)
		seq = np.random.permutation(m)
		pairs = seq.reshape(-1,2)
		
		if method != 'random' and method != 'spatial' and method != 'hybrid' and method != 'mixed_spatial':
			raise ValueError('Invalid argument for the "method" parameter \
valid arguments are "random", "spatial", "hybrid" and "mixed_spatial"')
		
		elif method == 'spatial':
			population = breed_spatial_(
					self.population, self.n_clusters, pairs)
		
		elif method == 'random':
			population = breed_random_(
					self.population, self.n_clusters, cut_off, pairs)
		
		elif method == 'hybrid':
			population = breed_hybrid_(self.population, self.n_clusters, pairs)

		elif method == 'mixed_spatial':
			population = breed_mixed_spatial_(
					self.population, self.n_clusters, pairs)
		
		self.population = population
		return population

#	def breed(self, cut_off=0.2, method='random'):#		generation = []
#		seq = np.random.permutation(self.size) #Create random sequence
#		pairs = seq.reshape(-1,2) #Match pairs as per random sequence
#
#		if method != 'random' and method != 'spatial':
#			raise ValueError('Please pass in a valid argument for the method n\
#				   parameter. Accepted values are "random" & "spatial"')
#		elif method == 'random':
#			if cut_off >= 1:
#				raise ValueError('cut_off rate argument must be less than 1')
#
#			else:
#				n = int(cut_off * self.n_clusters)
#
#				#From every pair breed two offsprings
#				for pair in pairs:
#					x_chrom = [self.population[pair[0]][:n],
#							self.population[pair[1]][:n]]
#
#					y_chrom = [self.population[pair[1]][n:],
#							self.population[pair[0]][n:]]
#
#					child_one = np.concatenate((x_chrom[0],
#								 y_chrom[0]), axis=0)
#
#					child_two = np.concatenate((x_chrom[1],
#								 y_chrom[1]), axis=0)
#
#					self.population.append(child_one)
#					self.population.append(child_two)
#
#
#		elif method == 'spatial':
#			for pair in pairs:
#				x_chrom = self.population[pair[0]]
#				y_chrom = self.population[pair[1]]
#				distances = []
#				for x_vector in x_chrom:
#					for y_vector in y_chrom:
#						distances.append(dist.euclidean(x_vector, y_vector))
#				#xdist_mat = np.array(distances)
#				xdist_mat = np.reshape(distances,(-1, self.n_clusters))
#				ydist_mat = xdist_mat.transpose()
#				child_one = []
#				child_two = []
#				for i in range(self.n_clusters):
#					gene_one = (x_chrom[i] + y_chrom[np.argmin(
#							xdist_mat[i])]) * 0.5
#					child_one.append(gene_one)
#
#					gene_two = (y_chrom[i] + x_chrom[np.argmin(
#							ydist_mat[i])]) * 0.5
#					child_two.append(gene_two)
#
#				self.population.append(np.array(child_one))
#				self.population.append(np.array(child_two))
#
#		#self.population = np.array(self.population)
#		return self.population


