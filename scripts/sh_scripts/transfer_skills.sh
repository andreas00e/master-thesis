#!/bin/bash
#SBATCH --job-name=transfer_skills
#SBATCH --output=outputs/transfer_skills/output/training_%j.log
#SBATCH --error=outputs/transfer_skills/error/training_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=lrz-dgx-1-p100x8 
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

mkdir -p outputs/transfer_skills/output outputs/transfer_skills/error

source /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/ehrensberger/sim/bin/activate
cd /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/ehrensberger/master-thesis

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PWD:$PYTHONPATH"

python3 scripts/py_scripts/transfer_skills.py

transfer_skills
