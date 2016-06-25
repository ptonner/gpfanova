import gpfanova, data

def get(p,numIndividual,f=False,n=10,interaction=False):

	s = ""
	if interaction:
		s = "interaction_"
	pf = None
	if f:
		pf = "results/qtl/parameters_%s%d.csv"%(s,n)

	x,y,effect,labels = data.multiple_effects([2]*p,numIndividual,fullFactorial=False,helmert_convert=True)
	m = gpfanova.fanova.FANOVA(x,y,effect,interactions=interaction,parameter_file=pf,helmert_convert=True)

	if f is None:
		m.parameter_cache['y_sigma'] = -2
		m.parameter_cache['prior0_lengthscale'] = -1
		for i in range(p):
			m.parameter_cache['prior%d_lengthscale'%(i+1)] = np.random.normal(-.5,.3)
			m.parameter_cache['prior%d_sigma'%(i+1)] = np.random.normal(0,.3)
		m.y,_ = m.samplePrior()

	return m,x,y,effect,labels

if __name__ == "__main__":
	import sys, getopt
	import matplotlib.pyplot as plt

	opts,args = getopt.getopt(sys.argv[1:],'n:if')

	p = 1000
	ni = 100
	n = 10
	interaction=False
	f = False
	for o,a in opts:
		if o == '-n':
			n = int(a)
		if o =='-i':
			interaction=True
		if o == '-f':
			f = True

	m,_,_,_,_ = get(p,ni,f)

	for i in range(p):
		# plt.figure(figsize=(20,20))
		gpfanova.plot.plotSingleEffect(m,i,data=True,_mean=False,offset=False,alpha=.5,empirical=True,individual=True);
		plt.savefig("results/qtl/data_%d.png"%i,bbox_inches="tight",dpi=300)
		plt.close()

	s = ""
	if interaction:
		s = "interaction_"

	try:
		m.sample(n,10,random=True)
	except Exception,e:
		m.save("results/qtl/parameters_%s%d.csv"%(s,n))
		raise(e)

	m.save("results/qtl/parameters_%s%d.csv"%(s,n))
