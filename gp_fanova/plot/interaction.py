from meta import *

def plotInteraction(m,i,k,kv=None,function=False,data=False,derivative=False,**kwargs):

	if function:
		_plot_function(m,i,k,kv,**kwargs)
	if data:
		_plot_data(m,i,k,kv,**kwargs)
	if derivative:
		_plot_derivative(m,i,k,kv,**kwargs)

def _plot_function(m,i,k,kv,_mean=False,burnin=0,subplots=True,offset=False,labels=None,origin=False,relative=False,controlFixed=True,**kwargs):

	m0 = m.mk[i]
	m1 = m.mk[k]
	if relative:
		m0-=1
		m1-=1

	ncol = m0
	nrow = m1
	if not kv is None:
		nrow = 1

	ylim = (1e9,1e-9)

	if nrow*ncol > len(colors):
		_cmap = plt.get_cmap('spectral')

	for j in range(m0):
		if kv is None:
			for l in range(m1):

				if subplots:
					plt.subplot(nrow,ncol,j+l*m0+1)
					plt.title("%d,%d"%(j,l))
					if origin:
						plt.plot([m.x.min(),m.x.max()],[0,0],c='k',lw=3)

				if m0*m1 <= len(colors):
					c = colors[j+l*m.mk[i]]
				else:
					r = .4
					c = _cmap(r+(1-r)*(j+l*m.mk[k]+1)/(m.mk[i]*m.mk[k]+1))

				# samples = m.effect_samples(k,j)[burnin:,:]
				if relative:
					if controlFixed:
						samples = m.relativeInteraction(i,j+1,k,l+1)
					else:
						samples = m.relativeInteraction(i,j+1,k,l+1,j,l)
				else:
					samples = m.parameterSamples("(%s,%s)_(%d,%d)" %(m.effect_suffix(i),m.effect_suffix(k),j,l)).values[burnin:,:]
					if offset:
						samples += m.parameterSamples("%s_%d" %(m.effect_suffix(i),j)).values[burnin:,:]
						samples += m.parameterSamples("%s_%d" %(m.effect_suffix(k),l)).values[burnin:,:]

				mean = samples.mean(0)
				std = samples.std(0)

				ylim = (min(ylim[0],min(mean-2*std)),max(ylim[1],max(mean+2*std)))

				l = None
				if not labels is None and len(labels)>j:
					l = str(labels[j])
				plt.plot(m.x,mean,color=c,label=l)
				plt.fill_between(m.x[:,0],mean-2*std,mean+2*std,alpha=.2,color=c)
		else:
			if m0 <= len(colors):
				c = colors[j]
			else:
				r = .4
				c = _cmap(r+(1-r)*(j+1)/(m.mk[i]+1))

			if relative:
				if controlFixed:
					samples = m.relativeInteraction(i,j+1,k,l+1)
				else:
					samples = m.relativeInteraction(i,j+1,k,l+1,j,l)
			else:
				samples = m.parameterSamples("(%s,%s)_(%d,%d)" %(m.effect_suffix(i),m.effect_suffix(k),j,kv)).values[burnin:,:]
				if offset:
					samples += m.parameterSamples("%s_%d" %(m.effect_suffix(i),j)).values[burnin:,:]
					samples += m.parameterSamples("%s_%d" %(m.effect_suffix(k),kv)).values[burnin:,:]

			mean = samples.mean(0)
			std = samples.std(0)

			ylim = (min(ylim[0],min(mean-2*std)),max(ylim[1],max(mean+2*std)))

			l = None
			if not labels is None and len(labels)>j:
				l = str(labels[j])
			plt.plot(m.x,mean,color=c,label=l)
			plt.fill_between(m.x[:,0],mean-2*std,mean+2*std,alpha=.2,color=c)

		if not labels is None:
			plt.legend(loc="best")

	if subplots and kv is None:
		for j in range(m0):
			for l in range(m1):
				plt.subplot(nrow,ncol,j+l*m0+1)
				plt.ylim(ylim)
	elif origin:
		plt.plot([m.x.min(),m.x.max()],[0,0],c='k',lw=3)
