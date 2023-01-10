#!/bin/bash

pids=()
for i in {0..5}
do
    python train_offline.py --config=configs/kitchen_config.py --double=True --env_name=kitchen-mixed-v0 --grad_pen=False --lambda_gp=1 --max_clip=7 --seed=$i --temp=8 &

    pids+=( "$!" )
    sleep 5 # add 5 second delay
done

for pid in "${pids[@]}"; do
    wait $pid
done

# sleep 2d
