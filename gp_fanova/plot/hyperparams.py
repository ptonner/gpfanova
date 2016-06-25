from meta import *

def plotHyperparameters(m,iterative=False,histogram=False,correlative=False,logspace=True,burnin=0,*args,**kwargs):

	if iterative:
		_plot_hyper_params_iterative(m,logspace,burnin,*args,**kwargs)
	if histogram:
		_plot_hyper_params_histogram(m,logspace,burnin,*args,**kwargs)
	elif correlative:
		_plot_hyper_params_correlative(m,logspace,burnin,*args,**kwargs)

def _plot_param(m,param,logspace,burnin):
	if logspace:
		plt.plot(m.parameter_history.loc[burnin:,param])
	else:
		plt.plot(10**m.parameter_history.loc[burnin:,param])

	# l = m.parameter_history.shape[0]
	# plt.xlim(-l*.1,l+l*.1)
	plt.title(param,fontsize=20)

def _plot_param_correlation(m,param1,param2,logspace,burnin):
	if logspace:
		plt.scatter(m.parameter_history.loc[burnin:,param1],m.parameter_history.loc[burnin:,param2])
	else:
		plt.scatter(pow(10,m.parameter_history.loc[burnin:,param1]),pow(10,m.parameter_history.loc[burnin:,param2]))
	# plt.title(param,fontsize=20)

def _plot_hyper_params_correlative(m,logspace,burnin,*args,**kwargs):

	params = ['y_sigma']
	for i in range(len(m.priorGroups())):
		params.extend(m._priorParameters(i))

	s = len(params)

	for i in range(len(params)):
		plt.subplot(s,s,i*s+i+1)
		_plot_param(m,params[i],logspace,burnin)

		for j in range(i):
			plt.subplot(s,s,i*s+j+1)
			_plot_param_correlation(m,params[j],params[i],logspace,burnin)


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

	params = ['y_sigma']
	for i in range(len(m.priorGroups)):
		params.append(m._priorParameters(i))

	s = len(params)
	for i in range(len(params)):
		if params[i] == '':
			continue
		plt.subplot(s,2,i+1)
		plot_param(params[i])
