import os
import math
import json
import argparse

import gym
import ray
import matplotlib.pyplot as plt
import numpy as np
import ray.rllib.agents as agents

import temperature_simulator as temp_sim
import heating_controller_config
import baseline_policy
from heating_controller_train import HeatingEnv, HeatingEnvCreator

# Register HeatingEnv with ray reinforcement learning library
ray.tune.registry.register_env('HeatingEnv-v0', HeatingEnvCreator)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Apply policy in inference mode for heating control.')
parser.add_argument('trial_path', type=str, nargs=1,
                    help='Path to folder with policy configuration and checkpoint')
parser.add_argument('checkpoint_num', type=int, nargs=1,
                    help="Checkpoint to be loaded for inference")
parser.add_argument('--baseline', type=str, nargs=1, default=None,
                    help="Name of baseline policy to be applied for same parameters as in policy configuration")

args = parser.parse_args()
trial_path = args.trial_path[0]
checkpoint_num = args.checkpoint_num[0]
baseline = args.baseline

# Construct path to checkpoint trained with heating_controller_train.py
ckpt_path = os.path.join(trial_path, 'checkpoint_{}'.format(checkpoint_num))

if not os.path.isdir(ckpt_path):
    print("Specified checkpoint {} does not exist".format(checkpoint_num))
    exit()

ckpt_name = os.path.join(ckpt_path, 'checkpoint-{}'.format(checkpoint_num))

params_path = os.path.join(trial_path, 'params.json')

with open(params_path, 'r') as params_file:
    config = json.loads(params_file.read())

config['env_config'] = heating_controller_config.env_config_dict


ray.init()


env = HeatingEnv(config['env_config'])

if baseline is not None:
    if baseline[0] == 'RandomPolicy':
        agent = baseline_policy.RandomPolicy(env.observation_space, env.action_space, config)
    elif baseline[0] == 'HeatWhenTooColdPolicy':
        agent = baseline_policy.HeatWhenTooColdPolicy(env.observation_space, env.action_space, config)
    else:
        print('Trying to run inference with unknown baseline policy type')

else:
    if 'PPO' in trial_path:
        agent = agents.ppo.ppo.PPOTrainer(config=config, env="HeatingEnv-v0")
    elif 'DQN' in trial_path:
        agent = agents.dqn.dqn.DQNTrainer(config=config, env="HeatingEnv-v0")
    elif 'SAC' in trial_path:
        agent = agents.sac.sac.SACTrainer(config=config, env="HeatingEnv-v0")
    else:
        print('Trying to run inference with unknown policy type')

    agent.restore(ckpt_name)

num_repetitions = 25

total_rewards = []

for i in range(num_repetitions):
    done = False
    idx = 0
    steps = []
    Ti = []
    To = []
    Ttgt = []
    actions = []
    rewards = []
    obs = env.get_obs()
    while (not done):
        steps.append(idx)

        action = agent.compute_action(obs)
        actions.append(action)

        obs, rew, done, _ = env.step(action)

        Ttgt.append(obs[1])
        To.append(obs[2])
        Ti.append(obs[3])
        rewards.append(rew)
        idx += 1

    env.reset()

    if i == (num_repetitions - 1):
        plt.plot(steps, Ti, label='Ti')
        plt.plot(steps, To, label='To')
        plt.plot(steps, Ttgt, label='Ttgt')
        plt.plot(steps, actions, label='Actions')
        plt.plot(steps, rewards, label='Rewards')
        plt.legend()
        plt.show()

    total_rewards.append(sum(rewards[100:]) / len(rewards[100:]))

total_rewards_array = np.array(total_rewards)
rewards_mean = sum(total_rewards_array) / len(total_rewards_array)
rewards_std = math.sqrt(sum((total_rewards_array - rewards_mean)**2) / len(total_rewards_array))
print("Average reward achieved over", num_repetitions, ":")
print("\tMean:", rewards_mean)
print("\tFluctuations:", rewards_std)
