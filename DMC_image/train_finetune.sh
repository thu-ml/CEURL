#!/bin/bash

ALGO=$1
DOMAIN=$2 # walker_mass, quadruped_mass, quadruped_damping
GPU_ID=$3

if [ "$DOMAIN" == "walker_mass" ]
then
    ALL_TASKS=("walker_stand_mass" "walker_walk_mass" "walker_run_mass" "walker_flip_mass")
elif [ "$DOMAIN" == "quadruped_mass" ]
then
    ALL_TASKS=("quadruped_stand_mass" "quadruped_walk_mass" "quadruped_run_mass" "quadruped_jump_mass")
elif [ "$DOMAIN" == "quadruped_damping" ]
then
    ALL_TASKS=("quadruped_stand_damping" "quadruped_walk_damping" "quadruped_run_damping" "quadruped_jump_damping")
else
    ALL_TASKS=()
    echo "No matching tasks, you can only take DOMAIN as walker_mass, quadruped_mass, or quadruped_damping"
    exit 0
fi

echo "Experiments started."
for seed in $(seq 0 2)
do
  export MUJOCO_EGL_DEVICE_ID=${GPU_ID}
  python dreamer_pretrain.py configs=dmc_pixels configs/agent=${ALGO} domain=${DOMAIN} seed=$seed device=cuda:${GPU_ID}
  for string in "${ALL_TASKS[@]}"
  do
      export MUJOCO_EGL_DEVICE_ID=${GPU_ID}
      python dreamer_finetune.py configs=dmc_pixels configs/agent=${ALGO} domain=${DOMAIN} seed=$seed device=cuda:${GPU_ID} snapshot_ts=2000000 finetune_domain=$string
  done
done
echo "Experiments ended."

# e.g.
# ./finetune.sh peac_lbs walker_mass 0