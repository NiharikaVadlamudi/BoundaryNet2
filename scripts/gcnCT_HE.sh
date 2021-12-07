#!/bin/bash
#SBATCH -A research
#SBATCH --job-name='V3_Rev_GCN_CT'
#SBATCH --qos=medium
#SBATCH --gres=gpu:2
#SBATCH  -n 16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output='AGCN/outputText/ver3_rev_ct.txt'
#SBATCH --mail-user=niharika.vadlamudi@research.iiit.ac.in
#SBATCH --mail-type=END

python3 REV3.py --title "v3_ct_he_rev" --hd_weight_file 'MCNN_HD_Train.json'  --exp experiments/gcn_cir_bi.json --expdir AGCN/v3_rev_circulum/  --decrement 0.6 --gamma 150                                                
