#!/bin/bash

pids=()
for i in {0..5}
do
   python train_offline.py --config=configs/mujoco_config.py --double=True --env_name=walker2d-medium-replay-v2 --grad_pen=False --lambda_gp=1 --max_clip=5 --seed=$i --temp=5 &

    pids+=( "$!" )
    sleep 5 # add 5 second delay
done

for pid in "${pids[@]}"; do
    wait $pid
done

# sleep 2d
