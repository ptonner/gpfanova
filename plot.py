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

def plot_hyper_params(m,iterative=False,histogram=False,correlative=False,logspace=True,*args,**kwargs):

	if iterative:
		_plot_hyper_params_iterative(m,logspace,*args,**kwargs)
	if histogram:
		_plot_hyper_params_histogram(m,logspace,*args,**kwargs)
	elif correlative:
		_plot_hyper_params_correlative(m,logspace,*args,**kwargs)

def _plot_param(m,param,logspace=True):
	if logspace:
		plt.plot(m.parameter_history[param])
	else:
		plt.plot(10**m.parameter_history[param])

	l = m.parameter_history.shape[0]
	plt.xlim(-l*.1,l+l*.1)
	plt.title(param,fontsize=20)

def _plot_param_correlation(m,param1,param2,logspace=True):
	if logspace:
		plt.scatter(m.parameter_history[param1],m.parameter_history[param2])
	else:
		plt.scatter(10**m.parameter_history[param1],10**m.parameter_history[param2])
	# plt.title(param,fontsize=20)

def _plot_hyper_params_correlative(m,logspace,*args,**kwargs):

	params = ['mu_sigma','mu_lengthscale']
	for i in range(m.k):
		params += ['%s*_sigma'%m.EFFECT_SUFFIXES[i],
				   '%s*_lengthscale'%m.EFFECT_SUFFIXES[i]]

	s = len(params)

	for i in range(len(params)):
		plt.subplot(s,s,i*s+i+1)
		_plot_param(m,params[i],logspace)

		for j in range(i):
			plt.subplot(s,s,i*s+j+1)
			_plot_param_correlation(m,params[i],params[j],logspace)


def _plot_hyper_params_iterative(m,logspace,*args,**kwargs):
	def plot_param(param):
		if logspace:
			plt.plot(m.parameter_history[param])
		else:
			plt.plot(10**m.parameter_history[param])
		plt.title(param,fontsize=20)

	nrows = 1 + m.k

	plt.subplot(nrows,2,1)
	plot_param('mu_sigma')
	plt.subplot(nrows,2,2)
	plot_param('mu_lengthscale')

	for i in range(m.k):
		plt.subplot(nrows,2,2+2*i+1)
		plot_param('%s*_sigma'%m.EFFECT_SUFFIXES[i])
		plt.subplot(nrows,2,2+2*i+2)
		plot_param('%s*_lengthscale'%m.EFFECT_SUFFIXES[i])

def _plot_hyper_params_histogram(m,logspace,*args,**kwargs):
	def plot_param(param):
		if logspace:
			plt.hist(m.parameter_history[param])
		else:
			plt.hist(10**m.parameter_history[param])
		plt.title(param,fontsize=20)

	nrows = 1 + m.k

	plt.subplot(nrows,2,1)
	plot_param('mu_sigma')
	plt.subplot(nrows,2,2)
	plot_param('mu_lengthscale')

	for i in range(m.k):
		plt.subplot(nrows,2,2+2*i+1)
		plot_param('%s*_sigma'%m.EFFECT_SUFFIXES[i])
		plt.subplot(nrows,2,2+2*i+2)
		plot_param('%s*_lengthscale'%m.EFFECT_SUFFIXES[i])
