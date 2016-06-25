from meta import *

def plotMultipleEffects(m,function=False,data=False,derivative=False,**kwargs):

	if function:
		_plot_function(m,i,k,**kwargs)
	if data:
		_plot_data(m,i,k,**kwargs)
	if derivative:
		_plot_derivative(m,i,k,**kwargs)

def _plot_function(m,_mean=False,offset=False,interactions=False,burnin=0,labels=None,**kwargs):

	if m.mk[k] > len(colors):
		_cmap = plt.get_cmap('spectral')

	nrow = m.k
	ncol = max(m.mk)

	for i in range(m.k):
		for j in range(m.mk[i]):
			plt.subplot(nrow,ncol,sum([mk*ncol for mk in m.mk[:i]])+j+1)
			plt.title("%d,%d"%(i,j))
			samples = m.parameter_history[m.effectIndexToCache(i,j)].values[burnin:,:]

			if offset:
				samples += m.parameter_history[m.functionIndex(0)].values[burnin:,:]

			mean = samples.mean(0)
			std = samples.std(0)

			l = None
			if not labels is None and len(labels)>j:
				l = str(labels[j])
			plt.plot(m.x,mean,color=c,label=l)
			plt.fill_between(m.x[:,0],mean-2*std,mean+2*std,alpha=.2,color=c)

		# if _mean:
		# 	mean = m.parameter_history[m.functionIndex(0)].values[burnin:,:].mean(0)
		# 	std = m.parameter_history[m.functionIndex(0)].values[burnin:,:].std(0)
		#
		# 	plt.plot(m.x,mean,'k')
		# 	plt.fill_between(m.x[:,0],mean-2*std,mean+2*std,alpha=.2,color='k')

		if not labels is None:
			plt.legend(loc="best")
