import matplotlib.pyplot as plt

# _colors = [u'b', u'g', u'r', u'c', u'm', u'y']
# _cmap = plt.get_cmap("Spectral")

colors = [u'b', u'g', u'r', u'c', u'm', u'y']

def plotSamplesEmpirical(x,samples,c='b',alpha=.2,interval=.9):
	if x.ndim > 1:
		x = x[:,0]

	samples.sort(1)
	mean = samples.mean(1)

	thresh = (1.-interval)/2
	li = int(thresh*samples.shape[1])
	ui = int((thresh+interval)*samples.shape[1])

	# std = samples.std(1)
	plt.plot(x,mean,color=c)
	plt.fill_between(x,samples[:,li],samples[:,ui],alpha=alpha,color=c)

	return samples[:,li],samples[:,ui]

def plotSamplesActual(x,samples,c='b',alpha=.2,colors=None,**kwargs):
	if x.ndim > 1:
		x = x[:,0]

	# samples.sort(1)
	# mean = samples.mean(1)

	# thresh = (1.-interval)/2
	# li = int(thresh*samples.shape[1])
	# ui = int((thresh+interval)*samples.shape[1])

	# std = samples.std(1)
	if colors is None:
		plt.plot(x,samples,color=c)
	else:
		for i in range(len(colors)):
			plt.plot(x,samples[:,i],color=colors[i],**kwargs)
	# plt.fill_between(x,samples[:,li],samples[:,ui],alpha=alpha,color=c)

	# return samples[:,li],samples[:,ui]
