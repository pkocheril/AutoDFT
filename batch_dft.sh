#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=167:59:59   # walltime; max. 168 hours
#SBATCH --ntasks=32   # number of processor cores (i.e. tasks); max. 32
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=6G   # memory per CPU core; max 6 GB/core (192 GB total)
#SBATCH -J "AutoDFT"   # job name
#SBATCH --mail-user=pkocheri@caltech.edu   # email address; update as needed
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

python auto_DFT.py --cores $(nproc) --mem $(nmem) --deuterate F --jobtype raman
# python full_FCC_batch.py

