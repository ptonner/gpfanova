import numpy as np

class Interval(object):

	def __init__(self,samples,alpha,ndim=1):
		self.samples = samples
		self.n = self.samples.shape[0]
		self.alpha = alpha

		if self.alpha < 0 or self.alpha > 1:
			raise ValueError("must provide alpha between 0 and 1")

		if self.samples.ndim != ndim:
			raise ValueError("sample dimensions does not match %d"%ndim)

	def contains(self,x):
		raise NotImplemented("implement this for your interval!")

class ScalarInterval(Interval):

	def __init__(self,samples,alpha,pi):
		Interval.__init__(self,samples,alpha,ndim=1)

		# method for computing p(theta | data)
		self.pi = pi

		# compute the hpd interval
		# Adapted from "Monte Carlo Estimation of Bayesian Credible and HPD Intervals", Ming-Hui Chen and Qi-Man Shao, 1999
		self.epsilon = [self.pi(t) for t in self.samples]
		self.epsilon.sort()
		j = int(self.n*self.alpha)
		self.epj = self.epsilon[j] # any observation with pi(x|D) > epj is in the region

	def contains(self,x):
		return self.pi(x) > self.epj

	def plot(self,lims,x=None):

		import matplotlib.pyplot as plt

		on = False
		regs = []
		for z in np.linspace(*lims):
			if not on and self.contains(z):
				on = True
				start = z
			elif on and not self.contains(z):
				on = False
				regs.append((start,z))

		for r in regs:
			plt.hlines(0,r[0],r[1],lw=30)

		if not x is None:
			plt.scatter(x,-.001,c='r',marker='x',s=50)
		plt.yticks([])
		plt.ylim(-.0015,.0005)


def functionInterval(samples,start=1e-6,alpha=.95,tol=1e-6,maxiter=100):

	# change alpha to be best possible given the number of samples
	p = samples.shape[0]
	alphaPotential = 1.*np.arange(p)/p
	alpha = filter(lambda x: abs(x-alpha)==abs(alpha-alphaPotential).min(),alphaPotential)[0]

	mean = samples.mean(0)
	std = samples.std(0)
	lb,ub = mean-2*std,mean+2*std

	epsilon = start

	# function to calculate the empirical interval alpha for a given epsilon, x
	check = lambda x: 1.*sum((samples>lb-x).all(1) & (samples<ub+x).all(1))/samples.shape[0]

	bounds = []

	# double to find lower and upper epislon bound
	while check(epsilon)<alpha:
		bounds.append((epsilon,check(epsilon)))
		epsilon *= 2

	bounds.append((epsilon,check(epsilon)))

	# binary search
	eLb,eUb = epsilon/2,epsilon
	i = 0
	while True:
		if abs(check(eLb)-alpha)<tol:
			epsilon = eLb
			break
		elif abs(check(eUb)-alpha)<tol:
			epsilon = eUb
			break

		nb = (eLb + eUb)/2

		if check(nb)<alpha:
			eLb = nb
			bounds.append((nb,check(nb)))
		else:
			eUb = nb
			bounds.append((nb,check(nb)))

		i+=1
		if i > maxiter:
			epsilon = eLb
			break

	return epsilon,bounds
