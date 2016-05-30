from kernel import Kernel
import numpy as np

class RBF(Kernel):
	"""Radical basis fuction"""

	@staticmethod
	def dist(X,lengthscale):
		X = X/lengthscale

		Xsq = np.sum(np.square(X),1)
		r2 = -2.*np.dot(X, X.T) + Xsq[:,None] + Xsq[None,:]
		r2 = np.clip(r2, 0, np.inf)
		return np.sqrt(r2)

	def _K(self,X,sigma,lengthscale):

		dist = RBF.dist(X,lengthscale)

		return sigma*np.exp(-.5*dist**2)
