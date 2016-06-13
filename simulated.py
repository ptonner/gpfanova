import gp_fanova, data, pyDOE

def model(mk,n,interaction,f=False):
	import numpy as np
	np.random.seed(123)

	x = np.linspace(-1,1)[:,None]
	# y = np.zeros((50,(e1+e2)*n))
	effect = np.array(pyDOE.fullfact(mk).tolist()*n).astype(int)
	y = np.zeros((50,effect.shape[0]))

	m = gp_fanova.fanova.FANOVA(x,y,effect,interactions=interaction)
	y,f_sample = m.sample_prior()

	if f:
		st = ""
		if interaction:
			st = "_interaction"
		f = "results/simulation_%d,%d_%d%s.csv"%(e[0],e[1],n,st)
	else:
		f = None
	m = gp_fanova.fanova.FANOVA(x,y,effect,interactions=interaction,parameter_file=f)

	return m,x,y,effect,f_sample

if __name__ == "__main__":
	import sys, getopt

	opts,args = getopt.getopt(sys.argv[1:],'n:s:e:i')

	s = 10 # number of samples from posterior
	n = 3 # number of obesrvations for each condition
	e = (2,2) # number of effect levels for each effect
	interaction=False # include interaction?

	for o,a in opts:
		if o == '-n':
			n = int(a)
		if o == '-s':
			s = int(a)
		if o == '-e':
			e = split(a)
			e = [int(t) for t in e]
		if o =='-i':
			interaction=True

	m,_,_,_,_ = model(e,n,interaction,False)
	m.sample(s,1,random=True)

	st = ""
	if interaction:
		st = "_interaction"
	# m.parameter_history.to_csv("results/simulation_%d,%d_%d_%s%d.csv"%(e[0],e[1],n,st,s),index=False)
	m.save("results/simulation_%d,%d_%d%s.csv"%(e[0],e[1],n,st))
