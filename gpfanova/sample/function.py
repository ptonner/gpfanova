from . import Sampler
import scipy
import numpy as np
from .. import linalg

class Function(Sampler):

	#def __init__(self,name,parameters,base,f,kernel):
		#Sampler.__init__(self,name,'Function',parameters,current_param_dependent=False)

	def __init__(self,name,x,base,f,kernel):
		self.base = base
		self.x = x
		self.f = f
		self.kernel = kernel

		# if name is None:
		# 	params = ['f%d(%s)'%(self.f,str(z)) for z in self.x]
		# else:
		params = ['%s(%s)'%(name,str(z)) for z in self.x]

		Sampler.__init__(self,name,'Function',params,current_param_dependent=False)

	def _sample(self):

		m = self.base.functionResidual(self.f)

		# in order to isolate the function f from the likelihood of each observation
		# we have to divide by the design matrix coefficient, which leaves a
		# multiplication to balance (which happens twice, e.g. squared).

		# get the design matrix coefficient of each observtion for this function
		# squared because it appears twice in the normal pdf exponential
		n = np.power(self.base.designMatrix[:,self.f],2)
		n = n[n!=0]

		missingValues = np.isnan(m)

		# scale each residual contribution by its squared dm coefficient
		m = np.nansum((m.T*n).T,0)

		# sum for computing the final covariance
		n = np.sum(((~missingValues).T*n).T,0)

		y_inv = self.base.y_k.K_inv(self.x)
		f_inv = self.kernel.K_inv(self.x)

		A = n*y_inv + f_inv
		b = np.dot(y_inv,m)

		# chol_A = np.linalg.cholesky(A)
		chol_A = linalg.jitchol(A)
		chol_A_inv = np.linalg.inv(chol_A)
		A_inv = np.dot(chol_A_inv.T,chol_A_inv)

	 	mu,cov = np.dot(A_inv,b), A_inv

		return scipy.stats.multivariate_normal.rvs(mu,cov)


class FunctionDerivative(Sampler):

	def __init__(self,name,d,parameters,base,f,kernel):
		Sampler.__init__(self,name,'Function',parameters,current_param_dependent=False)
		self.base = base
		self.d = d
		self.f = f
		self.kernel = kernel

	def _params(self):
		ka_inv = self.kernel.K_inv(self.base.x)
		obs = self.base.functionMatrix(only=[self.f])
		kb = self.kernel.dK(self.base.x,self.d)
		kba = self.kernel.dK(self.base.x,self.d,cross=True)

		mu = np.dot(kba,np.dot(ka_inv,obs))
		cov = kb - np.dot(kba,np.dot(ka_inv,kba.T))

		return mu,cov

	def _sample(self):

		mu,cov = self._params()

		return scipy.stats.multivariate_normal.rvs(mu[:,0],cov)
