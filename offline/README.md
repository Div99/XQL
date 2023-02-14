# Offline Reinforcement Learning with Extreme Q-Learning

## How to run the code

### Install dependencies

These are the same setup instructions as in [Implicit Q-Learning](https://github.com/ikostrikov/implicit_q_learning).

```bash
pip install --upgrade pip

pip install -r requirements.txt

# Installs the wheel compatible with Cuda 11 and cudnn 8.
pip install --upgrade "jax[cuda]>=0.2.27" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Also, see other configurations for CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

### Example training code

Locomotion
```bash
python train_offline.py --env_name=halfcheetah-medium-expert-v2 --config=configs/mujoco_config.py --max_clip=5 --sample_random_times=1 --temp=1
```

AntMaze
```bash
python train_offline.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000  --max_clip=5  --temp=0.8
```

Kitchen and Adroit
```bash
python train_offline.py --env_name=pen-human-v0 --config=configs/kitchen_config.py --max_clip=5 --sample_random_times=1 --temp=8
```

Finetuning on AntMaze tasks
```bash
python train_finetune.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_finetune_config.py --eval_episodes=100 --eval_interval=100000 --replay_buffer_size 2000000 --max_clip=5 --num_v_updates=4  --temp=0.8 --num_pretraining_steps=1000000
```

### Reproduction

For reproducing our experiments, please run the scripts in the [reproduce](reproduce) folder for the settings we use for each environment.

### 
This code was built on top of the IQL codebase [here](https://github.com/ikostrikov/implicit_q_learning).
