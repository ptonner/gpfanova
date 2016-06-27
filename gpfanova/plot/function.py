from meta import *
import numpy as np

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
