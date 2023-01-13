#!/bin/bash

pids=()
for i in {0..5}
do
   python train_offline.py --batch_size=256 --config=configs/mujoco_config.py --double=True --env_name=halfcheetah-medium-replay-v2 --max_clip=5 --temp=1 --seed=$i &

    pids+=( "$!" )
    sleep 5 # add 5 second delay
done

for pid in "${pids[@]}"; do
    wait $pid
done

# sleep 2d
