
if __name__ == "__main__":
	import argparse, sys, os
	sys.path.append("../..")
	import gpfanova, analysis


	parser = argparse.ArgumentParser(description='Run analysis of Cokol et al. data.')
	parser.add_argument('strains',metavar=('s'), type=str, nargs='*',
	                  help='strains to build model for')
	parser.add_argument('-n', dest='n_samples', action='store',default=10, type=int,
	                   help='number of samples to generate from posterior')
	parser.add_argument('-t', dest='thin', action='store',default=10, type=int,
	                   help='thinning rate for the posterior')
	parser.add_argument('--label', dest='label', action='store',default='', type=str,
	                   help='add a label to this run')
	parser.add_argument('--plates', dest='plates', action='store',default='', type=str,
	                   help='plates to use in this run')
	parser.add_argument('-i', dest='interactions', action='store_true',
	                   help='include interactions in the model')
	parser.add_argument('-g', dest='generateCommands', action='store_true',
	                   help='generate the commands for this script')
	parser.add_argument('-a', dest='analyze', action='store_true',
	                   help='analyze the output of this script')
	parser.add_argument('-s', dest='standard', action='store_true',
	                   help='analyze standard data')
	parser.add_argument('-p', dest='paraquat', action='store_true',
	                   help='analyze paraquat data')
	parser.add_argument('-o', dest='osmotic', action='store_true',
	                   help='analyze osmotic data')
	parser.add_argument('-e', dest='heatshock', action='store_true',
	                   help='analyze heatshock data')
	parser.add_argument('--helmertConvert', dest='helmertConvert', action='store_true',
	                   help='helmertConvert toggle for gpfanova')
	parser.add_argument('--scaleX', dest='scaleX', action='store_true',
	                   help='scaleX toggle for data')
	parser.add_argument('--batchEffects', dest='batchEffects', action='store_true',
	                   help='batchEffects toggle for data')
	parser.add_argument('--coprCompliment', dest='coprCompliment', action='store_true',
	                   help='run coprCompliment analysis')
	parser.add_argument('-m', dest='mean', action='store_true',
	                   help='convert data to mean')

	args = parser.parse_args()

	if args.generateCommands:
		print '\n'.join(generate_commands(args.n_samples,args.interactions))
	elif args.analyze:
		analyze()
	else:

		if not args.plates == '':
			plates = args.plates.split(",")

		if args.coprCompliment:

			x,y,effect,labels = analysis.data.hsalinarum_TF(['ura3','ura3+pMTFcmyc','VNG1179C+pMTFcmyc','VNG1179C-VNG1179C','copR'],
							standard=args.standard,paraquat=args.paraquat,osmotic=args.osmotic,heatshock=True,
							mean=args.mean,scaleX=args.scaleX,batchEffects=args.batchEffects,nanRemove=True,
							plates=['heatshock_12'])

			import numpy as np

			# columns: strain, ev, copr-vector
			neweffects = np.zeros((5,3),dtype=int)
			neweffects[labels.str.contains("VNG1179"),0] = 1
			neweffects[labels.str.contains("copR"),0] = 1
			neweffects[labels.str.contains("\+"),1] = 1
			neweffects[labels=='VNG1179C-VNG1179C',[1,2]] = 1
			effect = neweffects[effect[:,0],:]

		else:
			x,y,effect,_ = analysis.data.hsalinarum_TF(args.strains,standard=args.standard,paraquat=args.paraquat,osmotic=args.osmotic,heatshock=args.heatshock,mean=args.mean,scaleX=args.scaleX,batchEffects=args.batchEffects,nanRemove=True)

		m = gpfanova.fanova.FANOVA(x,y,effect,interactions=args.interactions,helmertConvert=args.helmertConvert)

		resultsDir = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir))


		s = ''
		if args.coprCompliment:
			s += "_coprCompliment"
		if args.interactions:
			s += '_interactions'
		if args.helmertConvert:
			s += "_helmertConvert"
		if args.mean:
			s+= '_mean'
		if args.scaleX:
			s+= '_scaleX'
		if args.batchEffects:
			s+= '_batchEffects'
		s+= "_"
		temp = ''
		if args.standard:
			s += temp + "standard"
			temp = '-'
		if args.paraquat:
			s += temp + "paraquat"
			temp = '-'
		if args.osmotic:
			s += temp + "osmotic"
			temp = '-'
		if args.heatshock:
			s += temp + "heatshock"
			temp = '-'

		if len(args.strains)>0:
			s += '_(%s)'%",".join(args.strains)

		if args.label!="":
			s += "_%s"%args.label

		nrestarts = 0
		while nrestarts < 10:
			try:
				m.sample(args.n_samples,args.thin)
			except Exception,e:
				m.save(os.path.join(resultsDir,'results/hsalTF/hsalTF%s.csv'%(s)))
				nrestarts+=1

				# try walking back the sampler
				m.parameter_cache = m.parameter_history.iloc[-1,:]

				print e

		m.save(os.path.join(resultsDir,'results/hsalTF/hsalTF%s.csv'%(s)))
