#!/bin/bash

pids=()
for i in {0..5}
do
   python train_offline.py --config=configs/mujoco_config.py --double=True --batch_size=256 --env_name=halfcheetah-medium-v2 --max_clip=7 --temp=2 --seed=$i &

    pids+=( "$!" )
    sleep 5 # add 5 second delay
done

for pid in "${pids[@]}"; do
    wait $pid
done

# sleep 2d
