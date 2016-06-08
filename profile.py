#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile

import data,gp_fanova

def profile(m,n=10):
	m.sample(n,1)

if __name__ == "__main__":
	import sys, getopt

	opt,args = getopt.getopt(sys.argv[1:],'n:')

	n = 10
	d = "beer"

	for o,a in opt:
		if o == '-n':
			n = int(a)

	if d == "beer":
		x,y,effect,_ = data.hsalinarum_beer_data()
	elif d == "strain":
		x,y,effect = data.hsalinarum_strain_data(['trmB','rosR'])
	m = gp_fanova.fanova.FANOVA(x,y,effect,hyperparam_kwargs={'y_sigma':(.1,10),'sigma':(.3,10),'lengthscale':(.3,10)})

	cProfile.runctx("profile(m)", globals(), locals(), "Profile.prof")

	s = pstats.Stats("Profile.prof")
	s.strip_dirs().sort_stats("time").print_stats()
