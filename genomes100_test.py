
if __name__ == "__main__":

	import sys, getopt
	import matplotlib
	matplotlib.use('Agg')

	import gp_fanova, data
	import matplotlib.pyplot as plt

	opts,args = getopt.getopt(sys.argv[1:],'n:v:l:')

	x,y,effect = data.one_effect_data(100)
	m = gp_fanova.fanova.FANOVA(x,y,effect,helmert_convert=True)
	m.parameter_cache['y_sigma'] = -1

	n = 1
	for o,a in opts:
		if o == '-n':
			n = int(a)
		elif o == '-v':
			v0,v1 = a.split(',')
			m.parameter_cache['prior0_sigma'] = float(v0)
			m.parameter_cache['prior1_sigma'] = float(v1)
		elif o == '-l':
			v0,v1 = a.split(',')
			m.parameter_cache['prior0_lengthscale'] = float(v0)
			m.parameter_cache['prior1_lengthscale'] = float(v1)

	m.y = m.sample_prior()


	plt.figure(figsize=(20,20))
	gp_fanova.plot.plot_single_effect(m,0,data=True,_mean=False,offset=False,alpha=1);
	plt.savefig("results/genomes100_data.png",bbox_inches="tight",dpi=300)

	m.sample(n,10,random=True)

	m.parameter_history.to_csv("results/genomes100_%d.csv"%n,index=False)
