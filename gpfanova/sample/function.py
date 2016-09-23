from . import Sampler
import scipy
import numpy as np
from .. import linalg

class Function(Sampler):

	def __init__(self,name,parameters,base,f,kernel):
		Sampler.__init__(self,name,'Function',parameters,current_param_dependent=False)
		self.base = base
		self.f = f
		self.kernel = kernel

	def _sample(self):

		m = self.base.functionResidual(self.f)
		# n = m.shape[0]
		# m = m.mean(0)

		# n = [np.power(self.base.design_matrix[i,self.f],2) for i in range(self.base.m)]
		n = np.power(self.base.design_matrix[:,self.f],2)
		n = n[n!=0]
		missingValues = np.isnan(m)
		m = np.nansum((m.T*n).T,0)
		n = np.sum(((~missingValues).T*n).T,0)

		y_inv = self.base.y_k.K_inv(self.base.x)
		f_inv = self.kernel.K_inv(self.base.x)

		A = n*y_inv + f_inv
		#b = n*np.dot(y_inv,m)
		b = np.dot(y_inv,m)

		chol_A = linalg.jitchol(A)
		chol_A_inv = np.linalg.inv(chol_A)
		A_inv = np.dot(chol_A_inv.T,chol_A_inv)

	 	mu,cov = np.dot(A_inv,b), A_inv

		return scipy.stats.multivariate_normal.rvs(mu,cov)


class FunctionDerivative(Sampler):

	def __init__(self,name,parameters,base,f,kernel):
		Sampler.__init__(self,name,'Function',parameters,current_param_dependent=False)
		self.base = base
		self.f = f
		self.kernel = kernel

	def _params(self):
		ka_inv = self.kernel.K_inv(self.base.x)
		obs = self.base.functionMatrix(only=[self.f])
		kb = self.kernel.dK(self.base.x,)
		kba = self.kernel.dK(self.base.x,cross=True)

		mu = np.dot(kba,np.dot(ka_inv,obs))
		cov = kb - np.dot(kba,np.dot(ka_inv,kba.T))

		return mu,cov

	def _sample(self):

		mu,cov = self._params()

		return scipy.stats.multivariate_normal.rvs(mu[:,0],cov)
