#!/bin/bash

# We provide default settings in `config` on all tasks for coresponding pipeline.
model="habi"
seeds="0 1 2 3 4"

declare -A pipeline_tasks
pipeline_tasks[mujoco]="halfcheetah-medium-expert-v2 halfcheetah-medium-replay-v2 halfcheetah-medium-v2 hopper-medium-expert-v2 hopper-medium-replay-v2 hopper-medium-v2 walker2d-medium-expert-v2 walker2d-medium-replay-v2 walker2d-medium-v2"
pipeline_tasks[kitchen]="kitchen-partial-v0 kitchen-mixed-v0"
pipeline_tasks[antmaze]="antmaze-medium-play-v2 antmaze-medium-diverse-v2 antmaze-large-play-v2 antmaze-large-diverse-v2"
pipeline_tasks[maze2d]="maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1"

# train
for pipeline in "${!pipeline_tasks[@]}"; do
for task in ${pipeline_tasks[$pipeline]}; do
for seed in $seeds; do
# generate goal direct behavior
python pipelines/${model}_d4rl_${pipeline}.py \
    mode=generate_goal_direct_behavior \
    task=$task \
    seed=$seed \
    enable_wandb=0
# habitize into habitual behavior
python pipelines/${model}_d4rl_${pipeline}.py \
    mode=train \
    task=$task \
    seed=$seed \
    enable_wandb=0
done
done
done