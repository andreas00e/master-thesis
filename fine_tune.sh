#!/bin/bash
#SBATCH --job-name=fine_tune
#SBATCH --output=training_%j.log
#SBATCH --error=training_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=lrz-hpe-p100x4
#SBATCH --nodelist=p100-001
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

source /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/ehrensberger/sim/bin/activate
cd /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/ehrensberger/master-thesis

python3 fine_tune.py

# 5781028