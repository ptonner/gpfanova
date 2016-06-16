import gp_fanova, data

def get(interaction,f=False,n=0):

	s = ""
	if interaction:
		s = "interaction_"
	pf = None
	if f:
		pf = "results/hsal_beer_%s%d.csv"%(s,n)

	x,y,effect,labels = data.hsalinarum_beer_data()
	m = gp_fanova.fanova.FANOVA(x,y,effect,interactions=interaction,parameter_file=pf)
	return m,x,y,effect,labels

if __name__ == "__main__":
	import sys, getopt

	opts,args = getopt.getopt(sys.argv[1:],'n:if')

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

	m,_,_,_,_ = get(interaction,f)

	s = ""
	if interaction:
		s = "interaction_"

	try:
		m.sample(n,10,random=True)
	except Exception,e:
		m.save("results/hsal_beer_%s%d.csv"%(s,n))
		raise(e)

	m.save("results/hsal_beer_%s%d.csv"%(s,n))
