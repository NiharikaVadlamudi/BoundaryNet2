#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --gres=gpu:2
#SBATCH -n 15
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output='AGCN/outputText/agcn_test.txt'
#SBATCH --mail-user=niharika.vadlamudi@research.iiit.ac.in
#SBATCH --mail-type=END

python3 agcn.py --mode 'test'  --expdir "GCN/test/" --exp "experiments/gcn_settings.json" --optfile "gcn_polygon_results.json" --metricfile "gcn_metrics.csv" --checkpoint <NUM>  --vis False 

