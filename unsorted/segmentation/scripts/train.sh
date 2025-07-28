#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 30G
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task 3
#SBATCH --job-name equiv_segm
#SBATCH --output segm.log

# load modules or conda environments here
module load cerebrum-hpc/1.0
module load anaconda/3.7
source /nafs/dtward/intelpython3/bin/activate

python ../segmentation_train.py --model unet --num_channels 16 --run 1005 --epochs 200 --label_flag /nafs/dtward/allen/rois/categories.csv --data_dir /nafs/dtward/allen/rois/64x64_sample_rate_0_8_all/

# /nafs/dtward/allen/rois/64x64_num_samples_120_all2