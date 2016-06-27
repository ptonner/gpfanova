

if __name__ == "__main__":
	import argparse, gpfanova, data

	parser = argparse.ArgumentParser(description='Run analysis of Cokol et al. data.')
	parser.add_argument('strains',metavar=('s'), type=str, nargs='*',
	                  help='strains to build model for')
	parser.add_argument('-n', dest='n_samples', action='store',default=10, type=int,
	                   help='number of samples to generate from posterior')
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

	args = parser.parse_args()

	if args.generateCommands:
		print '\n'.join(generate_commands(args.n_samples,args.interactions))
	elif args.analyze:
		analyze()
	else:
		strains = args.strains
		x,y,effect,_ = data.hsalinarum(hsalinarum_TF,standard=args.standard,paraquat=args.paraquat,osmotic=args.osmotic)
		m = gp_fanova.fanova.FANOVA(x,y,effect,interactions=args.interactions)

		s = ''
		if args.interactions:
			s = '_interactions'

		try:
			m.sample(args.n_samples,10)
		except Exception,e:
			m.save('results/hsalTF/%s-%s%s.csv'%(a1,a2,s))
			raise(e)

		m.save('results/hsalTF/%s-%s%s.csv'%(a1,a2,s))
