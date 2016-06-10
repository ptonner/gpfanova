
if __name__ == "__main__":
	import sys, getopt
	import gp_fanova, data

	opts,args = getopt.getopt(sys.argv[1:],'n:i')

	n = 10
	interaction=False
	for o,a in opts:
		if o == '-n':
			n = int(a)
		if o =='-i':
			interaction=True

	x,y,effect,_ = data.hsalinarum_beer_data()
	m = gp_fanova.fanova.FANOVA(x,y,effect,interaction=interaction)

	m.sample(n,1,random=True)

	s = ""
	if interaction:
		s = "interaction_"
	m.parameter_history.to_csv("results/hsal_beer_%s%d.csv"%(s,n),index=False)
