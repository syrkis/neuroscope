#!/bin/bash

#SBATCH --partition brown,red
#SBATCH --cpus-per-task=1
#SBATCH --time 06:00:00
#SBATCH --job-name jupyter-notebook
#SBATCH --output logs/jupyter.log
#SBATCH --gres=gpu:1              # :v100:1
#SBATCH --mem=120G
#SBATCH --account=students
# get tunneling info
XDG_RUNTIME_DIR=""
# Get the port numbers used by slurmctld and slurmd from slurm.conf
slurmctld_port=$(grep "^SlurmctldPort" /etc/slurm/slurm.conf | awk '{print $2}')
slurmd_port=$(grep "^SlurmdPort" /etc/slurm/slurm.conf | awk '{print $2}')

# Generate a random port number between 8000 and 9999
port=$(shuf -i8000-9999 -n1)

# Check if the generated port number is already in use
while netstat -atn | grep -q ":$port "; do
    # Generate a new random port number if the current one is already in use
    port=$(shuf -i8000-9999 -n1)
done
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel
ssh -N -L ${port}:${node}:${port} ${user}@hpc.itu.dk

Windows MobaXterm info
Forwarded port: ${port}
Remote server: ${node}
Remote port: ${port}
SSH server: hpc.itu.dk
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
module --ignore-cache load singularity/3.4.1
module --ignore-cache load CUDA/11.1.1-GCC-10.2.0

# DON'T USE ADDRESS BELOW.
# DO USE TOKEN BELOW
# TOKEN FROM ENVIRONMENT VARIABLE
singularity exec --nv container.sif jupyter lab --no-browser --port=${port} --ip=0.0.0.0
