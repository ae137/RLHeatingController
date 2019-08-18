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
import ray
import os

import temperature_simulator as temp_sim

# Register HeatingEnv with ray reinforcement learning library
ray.tune.registry.register_env('HeatingEnv-v0', HeatingEnvCreator)

To_min = 5      # Mean of minimum outside temperature in °C
To_max = 15     # Mean of maximum outside temperature in °C
T_target = 21

Tin_init_props = temp_sim.TempInsideInitProps(mean=T_target, spread=5)
Tout_init_props = temp_sim.TempOutsideInitProps(mean_min=To_min, spread_min=3,
                                                mean_max=To_max, spread_max=3)

# This configuration should be the same as in heating_controller_train.py
config = {
    'model': {
        "fcnet_hiddens": [32, 32],
        },
    "lr": 0.0005,
    "train_batch_size": 4096,
    "num_workers": 6,
    "env_config": {
        "h": 0.15,
        "l": 0.025,
        "temp_diff_penalty": 1,
        'horizon': 400,
        'temp_in_init': Tin_init_props,
        "out_temp_sim": temp_sim.EnvironmentSimulator(Tout_init_props),
        "target_temp": temp_sim.TargetTemperature(T_target)
        }
    }

# Construct path to checkpoint trained with heating_controller_train.py
base_path = os.environ['HOME'] + '/ray_results/demo'
trial_path = 'PPO_HeatingEnv_0_lr=0.0005_2019-08-18_10-32-28ewsqlec4'
checkpoint_num = 81

ckpt_path = base_path + '/' + trial_path + '/checkpoint_' + str(checkpoint_num) + \
    '/checkpoint-' + str(checkpoint_num)

ray.init()

agent = ppo.PPOAgent(config=config, env="HeatingEnv-v0")
agent.restore(ckpt_path)

steps = []
Ti = []
To = []
actions = []
rewards = []

done = False
idx = 0

env = HeatingEnv(config['env_config'])

obs = env.get_obs()


while (not done):
    steps.append(idx)

    action = agent.compute_action(obs)
    actions.append(action)

    obs, rew, done, _ = env.step(action)

    Ti.append(obs[0])
    To.append(obs[1])
    rewards.append(rew)
    idx += 1


plt.plot(steps, Ti, label='Ti')
plt.plot(steps, To, label='To')
plt.plot(steps, actions, label='Actions')
plt.plot(steps, rewards, label='Rewards')
plt.legend()
plt.show()
