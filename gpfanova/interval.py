import numpy as np

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
