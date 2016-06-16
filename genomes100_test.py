
if __name__ == "__main__":

	import sys, getopt
	import matplotlib
	matplotlib.use('Agg')

	import gp_fanova, data
	import matplotlib.pyplot as plt
	import numpy as np

	opts,args = getopt.getopt(sys.argv[1:],'n:v:l:')

	p = 20

	x,y,effect,_ = data.multiple_effects([3]*p,100,20,False,seed=True)
	m = gp_fanova.fanova.FANOVA(x,y,effect,helmert_convert=True)
	m.parameter_cache['y_sigma'] = -2
	m.parameter_cache['prior0_lengthscale'] = -3
	for i in range(p):
		m.parameter_cache['prior%d_lengthscale'%(i+1)] = np.random.uniform(-3,0)

	n = 1
	for o,a in opts:
		if o == '-n':
			n = int(a)
		# elif o == '-v':
		# 	v0,v1 = a.split(',')
		# 	m.parameter_cache['prior0_sigma'] = float(v0)
		# 	m.parameter_cache['prior1_sigma'] = float(v1)
		# elif o == '-l':
		# 	v0,v1 = a.split(',')
		# 	m.parameter_cache['prior0_lengthscale'] = float(v0)
		# 	m.parameter_cache['prior1_lengthscale'] = float(v1)

	m.y,_ = m.sample_prior()

	for i in range(p):
		# plt.figure(figsize=(20,20))
		gp_fanova.plot.plotSingleEffect(m,i,data=True,_mean=False,offset=False,alpha=.81,empirical=False,individual=True);
		plt.savefig("results/genomes100/data_%d.png"%i,bbox_inches="tight",dpi=300)
		plt.close()

	# m.sample(n,10,random=True)

	# m.parameter_history.to_csv("results/genomes100/parameters_%d_%d.csv"%(n,p),index=False)
