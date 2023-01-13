#!/bin/bash

pids=()
for i in {0..5}
do
   python train_offline.py --env_name=hopper-medium-expert-v2 --config=configs/mujoco_config.py --temp=2 --double=True --seed=$i --sample_random_times=1 &

    pids+=( "$!" )
    sleep 5 # add 5 second delay
done

for pid in "${pids[@]}"; do
    wait $pid
done

# sleep 2d
