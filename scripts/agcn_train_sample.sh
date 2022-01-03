#!/bin/bash
#SBATCH --job-name="AGCN Training"
#SBATCH -A niharika.v
#SBATCH -n 10
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output='AGCN/outputText/agcn_train.txt'
#SBATCH --mail-user=niharika.vadlamudi@research.iiit.ac.in
#SBATCH --mail-type=END

python3 agcn.py --mode 'train' --title "AGCN Training" --exp 'experiments/agcn_settings.json' --expdir 'AGCN/train/' --mcnn_weights 'weights/mcnn.pth' --training_type 'normal' 

echo "Finished [EXP3] Performance Based GCN Training "


