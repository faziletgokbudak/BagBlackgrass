#!/bin/bash
#SBATCH --account OZTIRELI-SL2-GPU
#SBATCH --partition pascal
#SBATCH -t 36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=BEGIN,FAIL,END

module purge
module load rhel7/default-gpu
module load cuda/10.2
module load cudnn/7.6_cuda-10.2

source /usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh

conda activate dsmil

python compute_feats.py # --dataset=/home/fg405/rds/hpc-work/IMAGE_ARCHIVE
