#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH  -n=15
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=teena.txt
#SBATCH --mail-user=niharika.vadlamudi@research.iiit.ac.in
#SBATCH --mail-type=END

python3 trainFinal.py --exp experiments/singleTestingMCNN.json --expdir palmiraWeighted3 --weighted_epoch "20"                                                            
