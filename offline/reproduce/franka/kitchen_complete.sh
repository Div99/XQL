#!/bin/bash

pids=()
for i in {0..5}
do
    python train_offline.py --config=configs/kitchen_config.py --double=True --env_name=kitchen-complete-v0 --max_clip=7 --seed=$i --temp=2 &

    pids+=( "$!" )
    sleep 5 # add 5 second delay
done

for pid in "${pids[@]}"; do
    wait $pid
done

# sleep 2d
