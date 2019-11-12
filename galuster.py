import numpy as np


class MeanGene:
	def __init__(self, n_clusters, n_variables):
		self.n_clusters = n_clusters
		self.n_variables = n_variables

		self.gene = np.random.random((self.n_clusters, self.n_variables))

	def __str__(self):
		return "This is the meanest gene ever!"

class VarGene:
	def __init__(self, n_variables, n_features):
		self.n_variables = n_variables
		self.n_features = n_features
		x = n_variables - n_features
		zeros = np.zeros(x)
		ones = np.ones(self.n_features)
		self.gene = np.hstack((zeros, ones))
		np.random.shuffle(self.gene)


	def __str__(self):
		return "This is the most variable gene ever!"

class Population:
	def __init__(self, size, g_type='means', **kwargs):
		self.size = size
		self.g_type = g_type
		self.pop = []

		for gene in range(self.size):

			if self.g_type != 'means' and self.g_type != 'variables':
				raise ValueError('Please input a valid gene type n\
					 Supported gene types are "means" and "variables"')
				break
			elif self.g_type == 'means':
				gene = MeanGene(kwargs['n_clusters'], kwargs['n_variables'])
				self.pop.append(gene.gene)
			elif self.g_type == 'variables':
				gene = VarGene(kwargs['n_variables'], kwargs['n_features'])
				self.pop.append(gene.gene)
			else:
				raise ValueError('Make sure you insert the appropriate kwargs')

	def __str__(self):
		return "This is the population"

	#def mutate(self, )