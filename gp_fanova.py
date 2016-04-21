import scipy
import GPy
import numpy as np

n = 50
p = 1
x = np.zeros((n,p))
x[:,0] = np.linspace(-1,1,n)

mu_mu = np.zeros(n)
mu_cov = GPy.kern.RBF(p).K(x)
mu_prior = scipy.stats.multivariate_normal(mu_mu,mu_cov,allow_singular=True)
_mu_sample = np.zeros((n,1))

def mu_sample():
	_mu_sample[:,0] = mu_prior.rvs()
	return _mu_sample

mu_sample()

ma = 2
alpha_mu = np.zeros(n)
alpha_cov = GPy.kern.RBF(p).K(x)
alpha_samples = np.zeros((n,ma))

def alpha_sample(i):
	if i == ma - 1:
		alpha_samples[:,i] = -np.sum(alpha_samples[:,:i],1)
		return alpha_samples[:,i]

	if i == 0:
		temp_mu = np.zeros(n)
	else:
		temp_mu = -np.sum(alpha_samples[:,:i],1) / (ma - i)
	temp_cov = 1.*(ma-i-1)/(ma-i) * alpha_cov

	alpha_samples[:,i] = scipy.stats.multivariate_normal.rvs(temp_mu,temp_cov)

	return alpha_samples[:,i]

for i in range(ma):
	alpha_sample(i)

y_cov = GPy.kern.RBF(p,variance=.01,lengthscale=2).K(x) + np.eye(n)*.001

def y_sample(i,s=1):

	return scipy.stats.multivariate_normal.rvs(_mu_sample[:,0]+alpha_samples[:,i],y_cov,size=s)

def plot_functions(**kwargs):
	import matplotlib.pyplot as plt

	plt.plot(alpha_samples,**kwargs)

	plt.gca().set_color_cycle(None)

	plt.plot(alpha_samples+_mu_sample,'--',**kwargs)

	plt.plot(_mu_sample,'k',**kwargs)
