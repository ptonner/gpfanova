from meta import *
import numpy as np
from .. import interval as _interval

def plotFunctionSamples(samples,x=None,c='b',alpha=.2,burnin=0,interval=True,intervalKwargs={}):

	if x is None:
		x = np.arange(samples.shape[1])
	elif x.ndim > 1:
		x = x[:,0]

	samples = samples[burnin:,:]
	mean = samples.mean(0)
	std = samples.std(0)

	epsilon = 0
	if interval:
		epsilon,_ = _interval.functionInterval(samples,**intervalKwargs)

	plt.plot(x,mean,color=c)
	plt.fill_between(x,mean-2*std,mean+2*std,alpha=alpha,color=c)

def plot_function(m,f,c='b',alpha=.2,burnin=0):

	samples = m.parameter_history[m.functionIndex(f)].values[burnin:,:]

	mean = samples.mean(0)
	std = samples.std(0)

	plt.plot(m.x,mean,color=c)
	plt.fill_between(m.x[:,0],mean-2*std,mean+2*std,alpha=alpha,color=c)

def plot_functions(m,f,c='b',alpha=.2,burnin=0):

	samples = np.sum([f[j] * m.parameter_history[m.functionIndex(j)].values[burnin:,:] for j in range(len(f))],0)

	mean = samples.mean(0)
	std = samples.std(0)

	plt.plot(m.x,mean,color=c)
	plt.fill_between(m.x[:,0],mean-2*std,mean+2*std,alpha=alpha,color=c)
