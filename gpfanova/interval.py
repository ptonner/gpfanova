
def functionInterval(samples,start=1e-6,alpha=.95,tol=1e-6):

	mean = samples.mean(0)
	std = samples.std(0)
	lb,ub = mean-2*std,mean+2*std

	epsilon = start

	# function to calculate the empirical interval alpha for a given epsilon, x
	check = lambda x: 1.*sum((samples>lb-x).all(1) & (samples<ub+x).all(1))/samples.shape[0]

	# double to find lower and upper epislon bound
	while check(epsilon)<alpha:
		epsilon *= 2

	# binary search
	eLb,eUb = epsilon/2,epsilon
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
		else:
			eUb = nb

	return epsilon
