from . import Sampler
import numpy as np

class Transform(Sampler):
	"""not a true 'sampler', but some variable that is a transform of other variables"""

	def __init__(self,name,parameters,base,index,transform):
		Sampler.__init__(self,name,'Transform',parameters,)
		# self.transform_fxn = transform_fxn
		self.base = base
		self.index = index
		self.transform = transform

	def _sample(self,*args,**kwargs):
		# return self.transform_fxn()
		return np.dot(self.base.functionMatrix(only=self.index),self.transform)
