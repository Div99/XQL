from jax.config import config
import os
from typing import Tuple

import collections
import datetime
import gym
import numpy as np
import tqdm
import time
import absl
import sys
import os
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
from dataclasses import dataclass

import wrappers
from dataset_utils import D4RLDataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner
import warnings
import imageio

# os.environ['MUJOCO_GL'] = 'egl'
FLAGS = flags.FLAGS

flags.DEFINE_string('load_dir', './tmp/', 'Checkpoint load path')
flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_path', 'temp.gif', 'Gif output path')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_float('temp', 1.0, 'Loss temperature')
flags.DEFINE_boolean('double', True, 'Use double q-learning')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('vanilla', False, 'Use vanilla RL training')
flags.DEFINE_integer('sample_random_times', 0, 'Number of random actions to add to smooth dataset')
flags.DEFINE_boolean('grad_pen', False, 'Add a gradient penalty to critic network')
flags.DEFINE_float('lambda_gp', 1, 'Gradient penalty coefficient')
flags.DEFINE_float('max_clip', 7., 'Loss clip value')
flags.DEFINE_integer('num_v_updates', 1, 'Number of value updates per iter')
flags.DEFINE_boolean('log_loss', True, 'Use log gumbel loss')

flags.DEFINE_boolean('noise', False, 'Add noise to actions')
flags.DEFINE_float('noise_std', 0.1, 'Noise std for actions')

config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)


@dataclass(frozen=True)
class ConfigArgs:
    sample_random_times: int
    grad_pen: bool
    noise: bool
    noise_std: float
    lambda_gp: int
    max_clip: float
    num_v_updates: int
    log_loss: bool


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env(env_name: str,
             seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def main(_):

    env = make_env(FLAGS.env_name, FLAGS.seed)

    kwargs = dict(FLAGS.config)

    args = ConfigArgs(sample_random_times=FLAGS.sample_random_times,
                      grad_pen=FLAGS.grad_pen,
                      lambda_gp=FLAGS.lambda_gp,
                      noise=FLAGS.noise,
                      max_clip=FLAGS.max_clip,
                      num_v_updates=FLAGS.num_v_updates,
                      log_loss=FLAGS.log_loss,
                      noise_std=FLAGS.noise_std)

    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    loss_temp=FLAGS.temp,
                    double_q=FLAGS.double,
                    vanilla=FLAGS.vanilla,
                    args=args,
                    **kwargs
                    )
    agent.load(FLAGS.load_dir)

    best_eval_returns = -np.inf
    stats = {'return': [], 'length': []}

    num_episodes = 5
    frames = []

    for _ in range(num_episodes):
        stats = collections.defaultdict(list)
        observation, done = env.reset(), False
        frame = env.render(mode='rgb_array')
        print(frame.shape)
        frames.append(frame)

        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
            frame = env.render(mode='rgb_array')
            frames.append(frame)

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    print("Finished executing", num_episodes, "episodes")
    print("Stats:", stats)

    save_path = FLAGS.save_path if FLAGS.save_path.endswith('.gif') else FLAGS.save_path + ".gif"

    print("Saving gif to:", save_path)
    frames = frames[::2]  # save every other frame
    imageio.mimsave(save_path, frames)

    # wandb.finish()
    env.close()
    sys.exit(0)

    raise SystemExit


if __name__ == '__main__':
    app.run(main)
