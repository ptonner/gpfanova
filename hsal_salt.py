
if __name__ == "__main__":
	import sys, getopt
	import gp_fanova, data

	opts,args = getopt.getopt(sys.argv[1:],'n:')

	n = 10
	for o,a in opts:
		if o == '-n':
			n = int(a)

	x,y,effect,_ = data.hsalinarum_beer_data()
	m = gp_fanova.fanova.FANOVA(x,y,effect)

	m.sample(n,1,random=True)

	m.parameter_history.to_csv("results/hsal_beer_%d.csv"%n,index=False)
