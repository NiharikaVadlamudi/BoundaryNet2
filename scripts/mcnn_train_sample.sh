#!/bin/bash
#SBATCH --job-name="MCNN Training"
#SBATCH -A niharika.v
#SBATCH --gres=gpu:2
#SBATCH -n 10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=MCNN/mcnn_train.txt
#SBATCH --mail-user=niharika.vadlamudi@research.iiit.ac.in
#SBATCH --mail-type=END

python3 mcnn.py --title 'MCNN Training'  --mode 'train' -exp 'experiments/mcnn_config.json' --expdir 'MCNN/train/' --training_type 'normal' --freeze_hd True --freeze_clf True --freeze_seg_branch False 


