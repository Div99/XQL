#!/bin/bash

pids=()
for i in {0..5}
do
    python train_offline.py --batch_size=256 --config=configs/antmaze_config.py --double=True --env_name=antmaze-medium-play-v0 --eval_episodes=100 --eval_interval=30000 --grad_pen=False --lambda_gp=0 --max_clip=5 --seed=$i --temp=0.8 &

    pids+=( "$!" )
    sleep 5 # add 5 second delay
done

for pid in "${pids[@]}"; do
    wait $pid
done

# sleep 2d
