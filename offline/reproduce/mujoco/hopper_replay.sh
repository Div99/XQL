#!/bin/bash

pids=()
for i in {0..5}
do
   python train_offline.py --batch_size=256 --config=configs/mujoco_config.py --double=True --env_name=hopper-medium-replay-v2 --temp=2 --seed=$i &

    pids+=( "$!" )
    sleep 5 # add 5 second delay
done

for pid in "${pids[@]}"; do
    wait $pid
done

# sleep 2d
