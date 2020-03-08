#!/bin/bash
#SBATCH --job-name=testing_munit
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --output=synth_output.txt
#SBATCH --error=synth_error.txt
#SBATCH --ntasks=1

module load anaconda/3

source $CONDA_ACTIVATE

conda activate ccaienv

cd /network/home/raghupas/SpadeGAN/scripts/

python train.py --config ../configs/mask_conditioning_HD_sim_lsgan.yaml
