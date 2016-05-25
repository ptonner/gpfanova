from . import Sampler
import scipy

class Gibbs(Sampler):

	def __init__(self,name,parameters,conditional_parameters_fxn,):
		Sampler.__init__(self,name,'Gibbs',parameters)
		self.conditional_parameters_fxn = conditional_parameters_fxn

	def _sample(self,*args,**kwargs):

		mu,cov = self.conditional_parameters_fxn(*args,**kwargs)
		sample = scipy.stats.multivariate_normal.rvs(mu,cov)
		return sample
