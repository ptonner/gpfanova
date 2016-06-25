from meta import *

def plotSingleEffect(m,k,function=False,data=False,derivative=False,**kwargs):

	if function:
		if type(function) == tuple and len(function)==3:
			plt.subplot(*function)
		_plot_single_effect_function(m,k,**kwargs)
	if data:
		if type(data) == tuple and len(data)==3:
			plt.subplot(*data)
		_plot_single_effect_data(m,k,**kwargs)
	if derivative:
		if type(derivative) == tuple and len(derivative)==3:
			plt.subplot(*derivative)
		_plot_single_effect_derivative(m,k,**kwargs)

def _plot_single_effect_derivative(m,k,subplots=None,_mean=False,offset=False,variance=False,burnin=None,**kwargs):
	if burnin is None:
		burnin = 0

	if subplots:
		ylim = (1e9,-1e9)

	for j in range(m.mk[k]):

		if m.mk[k] <= len(colors):
			c = colors[j]
		else:
			r = .4
			c = _cmap(r+(1-r)*(j+1)/(m.mk[k]+1))

		samples = m.effect_derivative(k,j,mean=offset)[burnin:,:]
		# if offset:
		# 	samples += m.derivative_history[m.mu_index()].values[burnin:,:]

		mean = samples.mean(0)
		std = samples.std(0)

		if subplots:
			plt.subplot(subplots[0],subplots[1],j+1)
			plt.plot([m.x[:,0].min(),m.x[:,0].max()],[0,0],'k',linewidth=3)
			ylim = (min(ylim[0],samples.min()),max(ylim[1],samples.max()))

		if variance:
			plt.plot(m.x,std,color=c)
		else:
			plt.plot(m.x,mean,color=c)
			plt.fill_between(m.x[:,0],mean-2*std,mean+2*std,alpha=.2,color=c)

			if subplots:
				plt.ylim(ylim)

	if _mean:
		mean = m.derivative_history[m.mu_index()].values[burnin:,:].mean(0)
		std = m.derivative_history[m.mu_index()].values[burnin:,:].std(0)

		if variance:
			plt.plot(m.x,std,color='k')
		else:
			plt.plot(m.x,mean,'k')
			plt.fill_between(m.x[:,0],mean-2*std,mean+2*std,alpha=.2,color='k')

def _plot_single_effect_function(m,k,subplots=None,_mean=False,offset=False,variance=False,burnin=None,labels=None,**kwargs):

	if burnin is None:
		burnin = 0

	if subplots:
		ylim = (1e9,-1e9)

	if m.mk[k] > len(colors):
		_cmap = plt.get_cmap('spectral')

	for j in range(m.mk[k]):

		if m.mk[k] <= len(colors):
			c = colors[j]
		else:
			r = .4
			c = _cmap(r+(1-r)*(j+1)/(m.mk[k]+1))
			
		samples = m.parameter_history[m.effectIndexToCache(k,j)].values[burnin:,:]
		if offset:
			samples += m.parameter_history[m.functionIndex(0)].values[burnin:,:]

		mean = samples.mean(0)
		std = samples.std(0)

		if subplots:
			plt.subplot(subplots[0],subplots[1],j+1)
			plt.plot([m.x[:,0].min(),m.x[:,0].max()],[0,0],'k',linewidth=3)
			ylim = (min(ylim[0],samples.min()),max(ylim[1],samples.max()))

		if variance:
			plt.plot(m.x,std,color=c)
		else:
			l = None
			if not labels is None and len(labels)>j:
				l = str(labels[j])
			plt.plot(m.x,mean,color=c,label=l)
			plt.fill_between(m.x[:,0],mean-2*std,mean+2*std,alpha=.2,color=c)

			if subplots:
				plt.ylim(ylim)

	if _mean:
		mean = m.parameter_history[m.functionIndex(0)].values[burnin:,:].mean(0)
		std = m.parameter_history[m.functionIndex(0)].values[burnin:,:].std(0)

		if variance:
			plt.plot(m.x,std,color='k')
		else:
			plt.plot(m.x,mean,'k')
			plt.fill_between(m.x[:,0],mean-2*std,mean+2*std,alpha=.2,color='k')

	if not labels is None:
		plt.legend(loc="best")

def _plot_single_effect_data(m,k,subplots=False,alpha=1,empirical=True,individual=False,interval=.9,**kwargs):

	if m.mk[k] > len(colors):
		_cmap = plt.get_cmap('spectral')

	if empirical:
		for j in range(m.mk[k]):

			if m.mk[k] <= len(colors):
				c = colors[j]
			else:
				r = .4
				c = _cmap(r+(1-r)*(j+1)/(m.mk[k]+1))

			samples = m.y[:,m.effect[:,k]==j]

			# sort each row
			samples.sort(1)
			mean = samples.mean(1)

			thresh = (1.-interval)/2
			li = int(thresh*samples.shape[1])
			ui = int((thresh+interval)*samples.shape[1])

			# std = samples.std(1)
			plt.plot(m.x,mean,color=c)
			plt.fill_between(m.x[:,0],samples[:,li],samples[:,ui],alpha=.2,color=c)
	elif individual:
		for i in range(m.y.shape[1]):

			if subplots:
				plt.subplot(subplots[0],subplots[1],m.effect[i,k]+1)

			j = m.effect[i,k]
			if m.mk[k] <= len(colors):
				c = colors[j]
			else:
				r = .4
				c = _cmap(r+(1-r)*(j+1)/(m.mk[k]+1))

			plt.plot(m.x,m.y[:,i],color=c,alpha=alpha)
	plt.ylim(m.y.min(),m.y.max())
