import matplotlib.pyplot as plt

_colors = [u'b', u'g', u'r', u'c', u'm', u'y',]
_cmap = plt.get_cmap("Spectral")

def plot_single_effect(m,k,function=False,data=False,derivative=False,**kwargs):

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

		if m.mk[k] <= len(_colors):
			c = _colors[j]
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

def _plot_single_effect_function(m,k,subplots=None,_mean=False,offset=False,variance=False,burnin=None,**kwargs):

	if burnin is None:
		burnin = 0

	if subplots:
		ylim = (1e9,-1e9)

	for j in range(m.mk[k]):

		if m.mk[k] <= len(_colors):
			c = _colors[j]
		else:
			r = .4
			c = _cmap(r+(1-r)*(j+1)/(m.mk[k]+1))

		samples = m.effect_samples(k,j)[burnin:,:]
		if offset:
			samples += m.parameter_history[m.mu_index()].values[burnin:,:]

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
		mean = m.parameter_history[m.mu_index()].values[burnin:,:].mean(0)
		std = m.parameter_history[m.mu_index()].values[burnin:,:].std(0)

		if variance:
			plt.plot(m.x,std,color='k')
		else:
			plt.plot(m.x,mean,'k')
			plt.fill_between(m.x[:,0],mean-2*std,mean+2*std,alpha=.2,color='k')

def _plot_single_effect_data(m,k,subplots=False,alpha=None,**kwargs):
	if alpha is None:
		alpha=1

	for i in range(m.y.shape[1]):

		if subplots:
			plt.subplot(subplots[0],subplots[1],m.effect[i,k]+1)

		c = _colors[m.effect[i,0]]

		plt.plot(m.x,m.y[:,i],color=c,alpha=alpha)
		plt.ylim(m.y.min(),m.y.max())
