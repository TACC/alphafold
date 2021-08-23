#!/bin/bash
# job.slurm
# -----------------------------------------------------------------
#SBATCH -J af2_eho                # Job name
#SBATCH -o af2_eho.%j.out         # Name of stdout output file
#SBATCH -e af2_eho.%j.err         # Name of stderr output file
#SBATCH -p normal                    # Queue (partition) name
#SBATCH -N 1                         # Total # of nodes
#SBATCH -n 1                         # Total # of mpi tasks
#SBATCH -t 14:00:00                  # Run time (hh:mm:ss)
#SBATCH -A SD2E-Community                 # Project/Allocation name
# -----------------------------------------------------------------

module unload xalt
module load tacc-singularity
module list

export SIF='/scratch/projects/tacc/bio/alphafold/images/alphafold_2.0.0.sif'
export AF2_HOME='/scratch/projects/tacc/bio/alphafold/'

singularity exec $SIF python3 run_alphafold.py --flagfile=$AF2_HOME/test-container/flags/reduced_dbs.ff \
	--fasta_paths=$AF2_HOME/test-container/input/sample.fasta \
    --output_dir=$SCRATCH/af2_reduced \
	--model_names=model_1