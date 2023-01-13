#!/bin/bash

pids=()
for i in {0..5}
do
    python train_offline.py --config=configs/antmaze_config.py --double=True --env_name=antmaze-large-play-v0 --eval_episodes=100 --eval_interval=30000 --seed=$i --max_clip=5  --temp=0.6 &

    pids+=( "$!" )
    sleep 5 # add 5 second delay
done

for pid in "${pids[@]}"; do
    wait $pid
done

# sleep 2d
