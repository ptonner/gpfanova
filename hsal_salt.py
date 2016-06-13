
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
	m = gp_fanova.fanova.FANOVA(x,y,effect,interactions=interaction)

	s = ""
	if interaction:
		s = "interaction_"

	try:
		m.sample(n,1,random=True)
	except Exception,e:
		m.save("results/hsal_beer_%s%d.csv"%(s,n))
		raise(e)

	m.save("results/hsal_beer_%s%d.csv"%(s,n))
