from base import Base
import numpy as np

class Mean(Base):

	def _build_design_matrix(self):
		return np.ones((self.m,1))

	def prior_groups(self):
		return [[0]]
