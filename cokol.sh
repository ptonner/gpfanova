#!/bin/bash
#SBATCH --job-name=cokol
#SBATCH --output=cokol.out
#SBATCH --mem=4000

python cokol.py $1 $2 $3 $4
