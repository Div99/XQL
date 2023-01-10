import argparse
import gym
import research # To run environment imports
import os

from research.datasets.replay_buffer import ReplayBuffer

parser = argparse.ArgumentParser()
parser.add_argument("--env", "-e", type=str, required=True)
parser.add_argument("--num-steps", '-n', type=int, required=True)
parser.add_argument("--path", '-p', type=str, required=True)

args = parser.parse_args()

assert os.path.exists(os.path.dirname(args.path)), "Dataset folder doesn't exist: " + str(args.path)
assert not os.path.exists(args.path), "Dataset already exists: " + str(args.path)

env = gym.make(args.env)
replay_buffer = ReplayBuffer(env.observation_space, env.action_space, capacity=args.num_steps, cleanup=False)

num_steps = 0
num_ep = -1
episode_length = 0
done = True
while num_steps < args.num_steps:
    if done:
        num_ep += 1
        print("Finished", num_ep, "episodes in", num_steps, "steps.")
        obs = env.reset()
        replay_buffer.add(obs)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # Determine the discount factor.
    if 'discount' in info:
        discount = info['discount']
    elif hasattr(env, "_max_episode_steps") and episode_length == env._max_episode_steps:
        discount = 1.0
    else:
        discount = 1 - float(done)

    # Store the consequences.
    replay_buffer.add(obs, action, reward, done, discount)
    num_steps += 1

print("Finished", num_ep, "episodes in", num_steps, "steps.")

replay_buffer.save(args.path)
