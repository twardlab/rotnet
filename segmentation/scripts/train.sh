#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 30G
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task 3
#SBATCH --job-name equiv_benchmark
#SBATCH --output segm.log

# load modules or conda environments here
module load cerebrum-hpc/1.0
module load anaconda/3.7
source /nafs/dtward/intelpython3/bin/activate

python ../segmentation_train.py --data_flag ... --label_flag ...
