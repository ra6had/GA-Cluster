import numpy as np
from sklearn.cluster import KMeans as km


class MeanChrom:
	def __init__(self, n_clusters, n_variables):
		self.n_clusters = n_clusters
		self.n_variables = n_variables

		self.chrom = np.random.random((self.n_clusters, self.n_variables))

	def __str__(self):
		return "This is the meanest chromosome ever!"

class VarChrom:
	def __init__(self, n_variables, n_features):
		self.n_variables = n_variables
		self.n_features = n_features
		x = n_variables - n_features
		zeros = np.zeros(x)
		ones = np.ones(self.n_features)
		self.chrom = np.hstack((zeros, ones)).astype(int)
		np.random.shuffle(self.chrom)


	def __str__(self):
		return "This is the most variable chromosome ever!"

class Population:
	def __init__(self, size, ch_type='means', **kwargs):
		self.size = size
		self.ch_type = ch_type
		self.pop = []

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
		return "This is the population"

	
#    def evaluate(self, X):
#        score = []
#        if self.ch_type == 'means':
#            n_clusters = len(self.pop[0])
#            for chromosome in self.pop:
#                kmeans = km(n_clusters, chromosome).fit(X)
#                centers = kmeans.cluster_centers_
#                clusters = predict(X)
#    
#    def select(self, prc=0.5):
#        
#        score = []
#        if self.ch_type == 'means':
#            for chromosome in self.pop:
#                
    