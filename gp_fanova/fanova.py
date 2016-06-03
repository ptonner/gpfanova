from base import Base
import numpy as np
from patsy.contrasts import Sum

class FANOVA(Base):

	EFFECT_SUFFIXES = ['alpha','beta','gamma','delta','epsilon']

	def __init__(self,x,y,effect):
		self.effect = effect
		self.k = self.effect.shape[1] # number of effects
		self.mk = [np.unique(self.effect[:,i]).shape[0] for i in range(self.k)] # number of levels for each effect

		self.contrasts = [self.effect_contrast_matrix(i) for i in range(self.k)]
		self.contrasts_interaction = {}
		for i in range(self.k):
			for j in range(i):
				self.contrasts_interaction[(j,i)] = np.kron(self.contrasts[j],self.contrasts[i])

		Base.__init__(self,x,y,fxn_names=self.fxn_names())

	def effect_contrast_matrix(self,i):
		h = Sum().code_without_intercept(range(self.mk[i])).matrix
		return h

	def fxn_names(self):
		fxn_names = {0:'mean'}

		ind = 1
		for i in range(self.k):
			for j in range(self.mk[i]-1):
				fxn_names[ind] = "%s*_%d" % (FANOVA.EFFECT_SUFFIXES[i],j)
				ind += 1

		for i in range(self.k):
			for j in range(self.mk[i]-1):
				for k in range(i+1,self.k):
					for l in range(self.mk[k]-1):
						fxn_names[ind] = "(%s,%s)*_(%d,%d)" % (FANOVA.EFFECT_SUFFIXES[i],FANOVA.EFFECT_SUFFIXES[k],j,l)
						ind += 1

		return fxn_names

	def effect_index(self,i,j):
		return 1 + sum(self.mk[:i]) + j

	def effect_interaction_index(self,i,j,k,l):
		if i > k:
			return self.effect_contrast_index(k,l,i,j)

		# mean + effects + interactions before i + interactions with i before k + i interactions before j + l
		return 1 + sum([m-1 for m in self.mk]) + \
					sum([[(self.mk[m]-1)*(self.mk[l]-1) for l in range(m,i)] for m in range(i)]) + \
					sum([(m-1)*(self.mk[i]-1) for m in self.mk[:k]]) + \
					(j-1)*(self.mk[k]-1) + l

	def prior_groups(self):
		g = [[0]]

		ind = 1
		for i in range(self.k):
			g.append(range(ind,ind+self.mk[i]-1))
			ind += self.mk[i]-1

		for i in range(self.k):
			for k in range(i+1,self.k):
				c = (self.mk[i]-1) * (self.mk[k]-1)
				g.append(range(ind,ind+c))
				ind += c

		return g

	def _build_design_matrix(self):
		d = 1
		for i in range(self.k):
			d += self.mk[i] - 1
			for j in range(i):
				d += (self.mk[i] - 1) * (self.mk[j] - 1)

		x = np.zeros((self.m,d))
		x[:,0] = 1

		for s in range(self.m):
			ind = 1
			for i in range(self.k):
				x[s,ind:ind+self.mk[i]-1] = self.contrasts[i][self.effect[s,i],:]
				ind += self.mk[i]-1

			for i in range(self.k):
				for j in range(i+1,self.k):

					z = self.effect[s,i] * self.mk[j] + self.effect[s,j]
					x[s,ind:ind+(self.mk[i]-1)*(self.mk[j]-1)] = self.contrasts_interaction[(i,j)][z,:]

		return x
