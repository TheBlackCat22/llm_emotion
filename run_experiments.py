import os
import time

text = """#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=RunMain
#SBATCH --time={}:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=32G
#SBATCH --output=logs/%j.log

ml purge
ml Miniconda3
ml WebProxy

cd $SCRATCH/Projects/emotion_alignment

source activate robust-dpo_env

echo "STARTING MAIN"
python train.py --config configs/{}.yaml --output_dir outputs/{} --dataset_path {}
"""

run_dict = [
    {
        'task' : 'dpo',
        'hours': 4,
        'dataset_path' : [
            'datasets/emotion-preference-anger',
            'datasets/emotion-preference-fear',
            'datasets/emotion-preference-joy',
            'datasets/emotion-preference-love',
            'datasets/emotion-preference-sadness'             
            ]
    }
]


slurm_files = []
for item in run_dict:
    for dataset_path in item['dataset_path']:

        output_dir = f"{item['task']}-{dataset_path.split('-')[-1]}"

        with open(f'slurm/{output_dir}.slurm', 'w') as f:
            f.write(text.format(item['hours'], item['task'], output_dir, dataset_path))
        slurm_files.append(f'{output_dir}.slurm')


time.sleep(10)
os.chdir('slurm')
for run_file in slurm_files:
    os.system(f'sbatch {run_file}')