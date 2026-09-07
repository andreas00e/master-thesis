#!/bin/bash

#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/ehrensberger/%x.%j.out

# Virtuelle Umgebung aktivieren
source /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/ehrensberger/sim/bin/activate

# In das Arbeitsverzeichnis wechseln
cd /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/train/

# Torchrun mit korrigierten Argumenten ausführen
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    --master_port=6045 \
    /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/ehrensberger/master-thesis/discover_skill.py
