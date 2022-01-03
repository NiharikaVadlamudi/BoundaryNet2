#!/bin/bash
#SBATCH --job-name="MCNN Testing"
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH -n 15
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=MCNN/mcnn_test.txt
#SBATCH --mail-user=niharika.vadlamudi@research.iiit.ac.in
#SBATCH --mail-type=END

python3 mcnn.py  --mode 'test' --exp 'experiments/mcnn_settings.json' --expdir 'MCNN/test/'  --modelfile 'weights/mcnn.pth' --optfile 'mcnn_polygon_file.json' --metricfile 'mcnn_results.csv' 
                                                      
