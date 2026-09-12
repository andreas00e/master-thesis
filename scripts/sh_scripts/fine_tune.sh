#!/bin/bash
#SBATCH --job-name=fine_tune
#SBATCH --output=outputs/fine_tune/output/training_%j.log
#SBATCH --error=outputs/fine_tune/error/training_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=lrz-dgx-1-p100x8 
#SBATCH --nodelist=dgx-001
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

source /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/ehrensberger/sim/bin/activate

python3 ../py_scripts/fine_tune.py