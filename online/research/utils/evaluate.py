import torch
import numpy as np
import collections

MAX_METRICS = {'success', 'is_success'}
LAST_METRICS = {}
MEAN_METRICS = {}

def eval_policy(env, model, num_ep):
    ep_rewards = []
    ep_lengths = []
    ep_metrics = collections.defaultdict(list)

    for i in range(num_ep):
        # Reset Metrics
        done = False
        ep_reward = 0
        ep_length = 0
        ep_metric = collections.defaultdict(list)
        
        obs = env.reset()
        while not done:
            with torch.no_grad():
                action = model.predict(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            for k, v in info.items():
                if isinstance(v, float) or np.isscalar(v):
                    ep_metric[k].append(v)
            
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)
        for k, v in ep_metric.items():
            if k in MAX_METRICS:
                ep_metrics[k].append(np.max(v))
            elif k in LAST_METRICS: # Append the last value
                ep_metrics[k].append(v[-1])
            elif k in MEAN_METRICS:
                ep_metrics[k].append(np.mean(v))
            else:
                ep_metrics[k].append(np.sum(v))

    metrics = dict(reward=np.mean(ep_rewards), stddev=np.std(ep_rewards), length=np.mean(ep_lengths))
    for k, v in ep_metrics.items():
        metrics[k] = np.mean(v)
    return metrics