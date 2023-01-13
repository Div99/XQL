#!/bin/bash

pids=()
for i in {0..5}
do
   python train_offline.py --batch_size=256 --config=configs/mujoco_config.py --double=True --env_name=walker2d-medium-expert-v2 --max_clip=5 --seed=$i --temp=2 &

    pids+=( "$!" )
    sleep 5 # add 5 second delay
done

for pid in "${pids[@]}"; do
    wait $pid
done