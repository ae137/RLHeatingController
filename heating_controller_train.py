""" heating_controller_train.py: Train a heating controller with reinforcement learning

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
import math
from gym.spaces import Discrete, Box

import ray
from ray.tune import run_experiments, grid_search

import temperature_simulator as temp_sim


def reward_comfort(target_temp, real_temp) -> float:
    """ Compute reward for temperature being close to target temperature

    Parameters
    ----------
    target_temp: float
        Current target temperature
    real_temp: float
        Current inside temperature

    Returns
    -------
    float
        Reward for temperature being close to target temperature
    """
    return -abs(target_temp - real_temp)**2


class HeatingEnv(gym.Env):
    """ OpenAI gym environment for training a heating controller with reinforcement learning """
    episode_counter = 0
    action_space = Discrete(2)
    observation_space = Box(-30, 50, shape=(2,), dtype=np.float32)

    def __init__(self, env_config):
        """ Initialize OpenAI gym environment for heating controller

        Parameters
        ----------
        env_config: dict
            Dictionary with configuration for gym environment
        """
        self.heater_strength = env_config["h"]                      # Heating strength
        self.ener_loss = env_config["l"]                            # Energy loss coefficient
        self.horizon = env_config['horizon']                        # Episode length
        self.temp_diff_penalty = env_config["temp_diff_penalty"]    # Weight of comfort reward

        self.out_temp_sim = env_config["out_temp_sim"]              # Outside temperature simulator
        self.target_temp = env_config["target_temp"]                # Target temperature simulator
        self.time = 0
        self.out_temp_sim.reset()

        self.temp_in_init_props = env_config['temp_in_init']        # Parameter for inside temp init
        self.state = self.get_init_state()

    def reset(self) -> np.array:
        """ Reset gym environment

        Returns
        -------
        np.array
            Observation of system state
        """
        HeatingEnv.episode_counter += 1
        self.time = 0

        self.out_temp_sim.reset()
        self.state = self.get_init_state()

        return self.get_obs()

    def step(self, action):
        """ Perform action in environment and return new system state

        Parameters
        ----------
        action: int
            Action from the policy to be executed in the environment
        Returns
        -------
        np.array
            New observation of system state
        float
            Reward for action
        bool
            Parameter signaling whether the episode is over
        {}
            Unused
        """
        self.time += 1
        Ti_old, To_old = self.get_obs()

        Ti_new = Ti_old + math.sqrt(max(Ti_old - To_old, 1)) * self.heater_strength * action \
            + (To_old - Ti_old) * self.ener_loss
        To_new = self.out_temp_sim.getOutTemp(self.time)

        self.update_state(Ti_new, To_new)

        rew = self.temp_diff_penalty * reward_comfort(self.target_temp.getTargetTemp(self.time),
                                                      Ti_new)

        return self.get_obs(), rew, (self.time > self.horizon), {}

    def get_init_state(self) -> np.array:
        """ Get initial state of system

        Returns
        -------
        np.array
            New state of system
        """
        return np.array([temp_sim.temp_inside_init(self.temp_in_init_props),
                         self.out_temp_sim.getOutTemp(self.time)])

    def update_state(self, Ti_new, To_new):
        """ Update system state with given temperatures

        Parameters
        -------
        Ti_new: float
            New inside temperature of system
        To_new: float
            New outside temperature of system
        """
        self.state = np.array([Ti_new, To_new])

    def get_obs(self):
        return self.state


def HeatingEnvCreator(env_config):
    """ Environment creator function

    Parameters
    ----------
    env_config: dict
        Environment configuration that is passed to OpenAI gym environment

    Returns
    -------
    HeatingEnv
        OpenAI gym environment for training reinforcement learning based heating controller
    """
    return HeatingEnv(env_config)


if __name__ == "__main__":
    To_min = 5      # Mean of minimum outside temperature in °C
    To_max = 15     # Mean of maximum outside temperature in °C
    T_target = 21   # Target temperature of system in °C

    Tin_init_props = temp_sim.TempInsideInitProps(mean=T_target, spread=5)
    Tout_init_props = temp_sim.TempOutsideInitProps(mean_min=To_min, spread_min=3,
                                                    mean_max=To_max, spread_max=3)

    ray.init()

    run_experiments({
        "demo": {
            "run": "PPO",       # This algorithm yielded the best performance so far
            # "run": "DQN",
            # "run": "A3C",
            "env": HeatingEnv,
            "stop": {
                "training_iteration": 100,
            },
            "checkpoint_freq": 9,
            "checkpoint_at_end": True,
            "config": {
                'model': {
                    # "use_lstm": True,
                    "fcnet_hiddens": [32, 32],
                    },
                "lr": grid_search([0.0005]),
                "train_batch_size": 4096,
                "num_workers": 6,       # 1,  # parallelism
                "env_config": {
                    "h": 0.15,
                    "l": 0.025,
                    "temp_diff_penalty": 1,
                    'horizon': 400,
                    'temp_in_init': Tin_init_props,
                    "out_temp_sim": temp_sim.EnvironmentSimulator(Tout_init_props),
                    "target_temp": temp_sim.TargetTemperature(T_target)
                    },
            },
        },
    })
