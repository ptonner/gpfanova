from base import Base
import numpy as np

class Mean(Base):

	def __init__(self,x,y):
		Base.__init__(self,x,y,fxn_names={0:'mean'})

	def _buildDesignMatrix(self):
		return np.ones((self.m,1))

	def priorGroups(self):
		return [[0]]
