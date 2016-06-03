from base import Base

class FANOVA(Base):

	def __init__(self,x,y,effect):
		self.effect = effect
		self.k = self.effect.shape[1] # number of effects
		self.mk = [np.unique(self.effect[:,i]).shape[0] for i in range(self.k)] # number of levels for each effect

		Base.__init__(self,x,y)

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
					z = self.effect_contrast_interaction_index(i,self.effect[s,i],j,self.effect[s,j])
					x[s,ind:ind+(self.mk[i]-1)*(self.mk[j]-1)] = self.contrasts_interaction[(i,j)][z,:]

		return x
