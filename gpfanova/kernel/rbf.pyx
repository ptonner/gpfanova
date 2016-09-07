from kernel import Kernel
import numpy as np

class RBF(Kernel):
	"""Radical basis fuction"""

	@staticmethod
	def dist(X,lengthscales):

		X = X/lengthscales

		Xsq = np.sum(np.square(X),1)
		r2 = -2.*np.dot(X, X.T) + Xsq[:,None] + Xsq[None,:]
		r2 = np.clip(r2, 0, np.inf)
		return np.sqrt(r2)

	def _K(self,X,sigma,*lengthscales):
		lengthscales = np.array(lengthscales)

		# print lengthscales

		dist = RBF.dist(X,lengthscales)
		return sigma*np.exp(-.5*dist**2)

	def _dK(self,X,cross,d,sigma,*lengthscales):

		k = self._K(X,sigma,*lengthscales)

		lengthscales = np.array(lengthscales)

		diff = np.zeros((X.shape[0],X.shape[0]))
		for i in range(X.shape[0]):
			for j in range(X.shape[0]):
				diff[i,j] = X[i,:] - X[j,:]

		if cross:
			return (-1./(lengthscales[d]))*diff*k

		return (1./(lengthscales[d]))*(1-(1./(lengthscales[d]))*(diff**2))*k
