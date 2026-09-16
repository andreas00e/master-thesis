#!/bin/bash
#SBATCH --job-name=fine_tune
#SBATCH --output=outputs/fine_tune/output/training_%j.log
#SBATCH --error=outputs/fine_tune/error/training_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=lrz-hpe-p100x4
#SBATCH --nodelist=p100-001
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

mkdir -p outputs/fine_tune/output outputs/fine_tune/error

source /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/ehrensberger/sim/bin/activate
cd /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/ehrensberger/master-thesis

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PWD:$PYTHONPATH"

python3 scripts/py_scripts/fine_tune.py