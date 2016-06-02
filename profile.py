#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile

import data,gp_fanova

x,y,effect = data.hsalinarum_strain_data(['trmB','rosR'])
m = gp_fanova.base.GP_FANOVA(x,y,effect)
def profile():
	m.sample(10,1)

cProfile.runctx("profile()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
