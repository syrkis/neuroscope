#!/bin/bash

#SBATCH --job-name=container       # Job name
#SBATCH --output=logs/build.log    # Name of output file
#SBATCH --cpus-per-task=1          # Schedule one core
#SBATCH --time=01:00:00            # Run time (hh:mm:ss)
#SBATCH --partition=red,brown      # Run on either the Red or Brown queue
#SBATCH --mem=32G                  # memory
#SBATCH --account=students         # account


module load singularity

singularity build container.sif docker://syrkis/neuroscope
