#!/bin/bash
DOMAIN=$1 # walker_mass, quadruped_mass, quadruped_damping
GPU_ID=$2
FINETUNE_TASK=$3

echo "Experiments started."
for seed in $(seq 0 9)
do
    export MUJOCO_EGL_DEVICE_ID=${GPU_ID}
    python finetune.py configs/agent=ddpg domain=${DOMAIN} seed=${seed} device=cuda:${GPU_ID} snapshot_ts=0 finetune_domain=${FINETUNE_TASK} num_train_frames=2000010
done
echo "Experiments ended."

# e.g.
# ./finetune_ddpg.sh walker_mass 0 walker_stand_mass