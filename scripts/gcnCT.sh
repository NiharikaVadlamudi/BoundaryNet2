#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --gres=gpu:2
#SBATCH  -n 16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output='AGCN/outputText/ct_eh.txt'
#SBATCH --mail-user=niharika.vadlamudi@research.iiit.ac.in
#SBATCH --mail-type=END

python3 core/AGCN/gcnCurriculumTrain.py --title "CT_EH" --hd_weight_file 'MCNN_HD_Train.json'  --exp experiments/gcn2.json --expdir curriculum/ --weighted_training False --weighted_epoch 0  --vis False --resume False                                                     
