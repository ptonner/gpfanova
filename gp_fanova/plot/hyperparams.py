from meta import *

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

	params = ['y_sigma','mu_sigma','mu_lengthscale']
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

	params = ['y_sigma','','mu_sigma','mu_lengthscale']
	for i in range(m.k):
		params += ['%s*_sigma'%m.EFFECT_SUFFIXES[i],
				   '%s*_lengthscale'%m.EFFECT_SUFFIXES[i]]

	s = 2 + m.k
	for i in range(len(params)):
		if params[i] == '':
			continue
		plt.subplot(s,2,i+1)
		plot_param(params[i])

	# nrows = 1 + m.k
	#
	# plt.subplot(nrows,2,1)
	# plot_param('mu_sigma')
	# plt.subplot(nrows,2,2)
	# plot_param('mu_lengthscale')
	#
	# for i in range(m.k):
	# 	plt.subplot(nrows,2,2+2*i+1)
	# 	plot_param('%s*_sigma'%m.EFFECT_SUFFIXES[i])
	# 	plt.subplot(nrows,2,2+2*i+2)
	# 	plot_param('%s*_lengthscale'%m.EFFECT_SUFFIXES[i])
