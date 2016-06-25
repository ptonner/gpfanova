from meta import *

def plotTwoEffect(m,i,k,function=False,data=False,derivative=False,**kwargs):

	if function:
		_plot_function(m,i,k,**kwargs)
	if data:
		_plot_data(m,i,k,**kwargs)
	if derivative:
		_plot_derivative(m,i,k,**kwargs)

def _plot_function(m,i,k,subplots=None,_mean=False,offset=False,interactions=False,burnin=0,labels=None,**kwargs):

	if m.mk[k] > len(colors):
		_cmap = plt.get_cmap('spectral')

	ncol = m.mk[i]
	nrow = m.mk[k]

	for j in range(m.mk[i]):
		for l in range(m.mk[k]):

			plt.subplot(nrow,ncol,j+l*m.mk[i]+1)

			if m.mk[i]*m.mk[k] <= len(colors):
				c = colors[j]
			else:
				r = .4
				c = _cmap(r+(1-r)*(j+l*m.mk[k]+1)/(m.mk[i]*m.mk[k]+1))

			samples = m.parameter_history[m.effectIndexToCache(i,j)].values[burnin:,:] + m.parameter_history[m.effectIndexToCache(k,l)].values[burnin:,:]
			if offset:
				samples += m.parameter_history[m.functionIndex(0)].values[burnin:,:]

			if interactions:
				pass

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
