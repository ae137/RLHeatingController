""" heating_controller_simulate.py: Simulate a trained heating controller

Copyright (C) 2017 Andreas Eberlein

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""

from heating_controller_train import HeatingEnv, HeatingEnvCreator

import matplotlib.pyplot as plt
import numpy as np
import gym
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
import ray
import os

import temperature_simulator as temp_sim
import heating_controller_config
import baseline_policy

# Register HeatingEnv with ray reinforcement learning library
ray.tune.registry.register_env('HeatingEnv-v0', HeatingEnvCreator)

# This configuration should be the same as in heating_controller_train.py
config = {
    'model': {
        "fcnet_hiddens": [8, 8],
        },
    "lr": 0.0005,
    "train_batch_size": 4096,
    "num_workers": 6,
    "env_config": heating_controller_config.env_config_dict
    }

# Construct path to checkpoint trained with heating_controller_train.py
base_path = os.environ['HOME'] + '/ray_results'
# trial_path = 'demo_PPO_2/PPO_HeatingEnv_4935f5bd_2020-02-01_00-21-409_2ik84_' # [32, 32]
trial_path = 'demo_PPO_2/PPO_HeatingEnv_4935f5b4_2020-01-31_23-13-38skhvlqqr'   # [8, 8]
checkpoint_num = 200

ckpt_path = base_path + '/' + trial_path + '/checkpoint_' + str(checkpoint_num) + \
    '/checkpoint-' + str(checkpoint_num)

ray.init()


env = HeatingEnv(config['env_config'])

# agent = baseline_policy.HeatWhenTooColdPolicy(env.observation_space, env.action_space, config)
# agent = baseline_policy.RandomPolicy(env.observation_space, env.action_space, config)

# agent = ppo.PPOAgent(config=config, env="HeatingEnv-v0")
agent = ppo.PPOAgent(config=config, env="HeatingEnv-v0")
agent.restore(ckpt_path)


steps = []
Ti = []
To = []
Ttgt = []
actions = []
rewards = []

done = False
idx = 0

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


plt.plot(steps, Ti, label='Ti')
plt.plot(steps, To, label='To')
plt.plot(steps, Ttgt, label='Ttgt')
plt.plot(steps, actions, label='Actions')
plt.plot(steps, rewards, label='Rewards')
plt.legend()
plt.show()

print("Average reward achieved over last", len(rewards) - 100, "steps:\t",
      sum(rewards[100:]) / (len(rewards) - 100))
