from . import Sampler
import scipy
import numpy as np

class Function(Sampler):

	def __init__(self,name,parameters,base,f,kernel):
		Sampler.__init__(self,name,'Function',parameters,current_param_dependent=False)
		self.base = base
		self.f = f
		self.kernel = kernel

	def _sample(self):

		m = self.base.function_residual(self.f)
		# n = m.shape[0]
		# m = m.mean(0)

		# n = [np.power(self.base.design_matrix[i,self.f],2) for i in range(self.base.m)]
		n = np.power(self.base.design_matrix[:,self.f],2)
		n = n[n!=0]
		m = np.sum((m.T*n).T,0)
		n = np.sum(n)

		y_inv = self.base.y_k.K_inv(self.base.x)
		f_inv = self.kernel.K_inv(self.base.x)

		A = n*y_inv + f_inv
		#b = n*np.dot(y_inv,m)
		b = np.dot(y_inv,m)

		chol_A = np.linalg.cholesky(A)
		chol_A_inv = np.linalg.inv(chol_A)
		A_inv = np.dot(chol_A_inv.T,chol_A_inv)

	 	mu,cov = np.dot(A_inv,b), A_inv

		return scipy.stats.multivariate_normal.rvs(mu,cov)
