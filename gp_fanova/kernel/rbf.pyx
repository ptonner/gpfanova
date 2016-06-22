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

	def _dK(self,X,cross,sigma,lengthscale):

		k = self._K(X,sigma,lengthscale)
		# dist = RBF.dist(X,lengthscale)
		# if cross:
		# 	return (-1./(lengthscale**2))*dist*k
		#
		# return (1./(lengthscale**2))*(1-(1./(lengthscale**2))*(dist**2))*k

		diff = np.zeros((X.shape[0],X.shape[0]))
		for i in range(X.shape[0]):
			for j in range(X.shape[0]):
				diff[i,j] = X[i,:] - X[j,:]

		if cross:
			return (-1./(lengthscale))*diff*k

		return (1./(lengthscale))*(1-(1./(lengthscale))*(diff**2))*k
		# 	return (-1./(lengthscale**2))*diff*k
		#
		# return (1./(lengthscale**2))*(1-(1./(lengthscale**2))*(diff**2))*k
