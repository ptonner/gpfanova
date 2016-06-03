from meta import *

def plot_function(m,f,c='b',alpha=.2,burnin=0):

	samples = m.parameter_history[m.function_index(f)].values[burnin:,:]

	mean = samples.mean(0)
	std = samples.std(0)

	plt.plot(m.x,mean,color=c)
	plt.fill_between(m.x[:,0],mean-2*std,mean+2*std,alpha=alpha,color=c)