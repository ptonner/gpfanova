from base import Base
import numpy as np
from prior import Prior

class Mean(Base):
	"""A simple model where all observations come from a mean with GP prior."""

	def __init__(self,x,y,*args,**kwargs):
		# Base.__init__(self,x,y,fxn_names={0:'mean'})
		Base.__init__(self,x,y,np.ones((y.shape[1],1)),[Prior([0],'Mean')],*args,**kwargs)

	def _buildDesignMatrix(self):
		return np.ones((self.m,1))

	def priorGroups(self):
		return [[0]]
