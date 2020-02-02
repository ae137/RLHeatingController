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
import random
import gym
import math
import gym.spaces

import ray
import ray.tune

import temperature_simulator as temp_sim
import heater_state_machine as heater_state
import heating_controller_config


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


def reward_action_change(last_action, current_action) -> float:
    """ Compute reward for changing the action between steps

    Parameters
    ----------
    last_action:    float
        Last action
    current_action: float
        Current action

    Returns
    -------
    float
        Reward for changing actions between steps
    """
    return -abs(last_action - current_action)


class HeatingEnv(gym.Env):
    """ OpenAI gym environment for training a heating controller with reinforcement learning """
    episode_counter = 0

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
        self.action_penalty = env_config["action_penalty"]          # Weight of action changes
        self.len_hist = env_config["len_hist"]                      # History length for inside temp
        self.last_action = 0                                        # Variable storing last action

        self.out_temp_sim = env_config["out_temp_sim"]              # Outside temperature simulator
        self.tgt_temp_sim = env_config["tgt_temp_sim"]              # Target temperature simulator
        self.heater = env_config["heater"]                          # Heating power simulator
        self.time = 0
        self.out_temp_sim.reset()
        self.tgt_temp_sim.reset()

        self.state = self.get_init_state()

        # Action space:
        #   0: Heating off
        #   1: Heating on
        self.action_space = gym.spaces.Discrete(2)
        # Observation space at t = 0:
        # Ttgt[4], Ttgt[0], Tout[0], Tin[0], Tin[-1], Tin[-2], Tin[-3]
        # Ttgt[4]: Target temperature four time steps in the future
        # Ttgt[0]: Current target temperature
        # Tout[0]: Outside temperature
        # Tin[-n]: Inside temperature n time steps ago
        self.observation_space = gym.spaces.Box(-30, 50,
                                                shape=(3 + self.len_hist,),
                                                dtype=np.float32)

    def reset(self) -> np.array:
        """ Reset gym environment

        Returns
        -------
        np.array
            Observation of system state
        """
        HeatingEnv.episode_counter += 1
        self.time = random.randint(0, 96)

        self.out_temp_sim.reset()
        self.tgt_temp_sim.reset()
        self.heater.reset()

        self.state = self.get_init_state()
        self.last_action = 0

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
        _, _, To_old, Ti_old, *_ = self.get_obs()

        Ti_new = Ti_old + math.sqrt(max(Ti_old - To_old, 1)) * self.heater_strength \
            * self.heater.on_event(action) + (To_old - Ti_old) * self.ener_loss
        To_new = self.out_temp_sim.getOutTemp(self.time)

        self.update_state(Ti_new, To_new)

        rew = self.temp_diff_penalty * reward_comfort(self.tgt_temp_sim.getTargetTemp(self.time),
                                                      Ti_new) \
            + self.action_penalty * reward_action_change(self.last_action, action)

        self.last_action = action
        return self.get_obs(), rew, (self.time > self.horizon), {}

    def get_init_state(self) -> np.array:
        """ Get initial state of system

        Returns
        -------
        np.array
            New state of system
        """
        T_tgt_4 = self.tgt_temp_sim.getTargetTemp(self.time + 4)
        T_tgt = self.tgt_temp_sim.getTargetTemp(self.time)
        To_init = self.out_temp_sim.getOutTemp(self.time)
        Ti_init = self.tgt_temp_sim.getTargetTemp(self.time) + random.uniform(-3, 3)
        return np.array([T_tgt_4, T_tgt, To_init, *([Ti_init]*self.len_hist)])

    def update_state(self, Ti_new, To_new):
        """ Update system state with given temperatures

        Parameters
        -------
        Ti_new: float
            New inside temperature of system
        To_new: float
            New outside temperature of system
        """
        old_state = self.state
        T_tgt_4 = self.tgt_temp_sim.getTargetTemp(self.time + 4)
        T_tgt = self.tgt_temp_sim.getTargetTemp(self.time)

        self.state = np.zeros_like(old_state)
        self.state[0:4] = np.array([T_tgt_4, T_tgt, To_new, Ti_new])
        if self.len_hist > 1:
            self.state[4:] = old_state[3:-1]

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


demo_PPO_config = {
    "run": "PPO",       # This algorithm yielded the best performance so far
    "env": HeatingEnv,
    "stop": {
        "training_iteration": 200,
    },
    "checkpoint_freq": 20,
    "checkpoint_at_end": True,
    "config": {
        'model': {
            "use_lstm": False, # ray.tune.grid_search([False, True]),
            "fcnet_hiddens": ray.tune.grid_search([[16, 16], [32, 32], [16, 16, 16], [32, 32, 32]]),
            },
        "lr": ray.tune.grid_search([0.0005, 0.001]),
        "train_batch_size": 4096,   # 8192
        "gamma": 0.97,
        "entropy_coeff": 0.01,
        # "vf_clip_param": 10.0,
        # "grad_clip": 10,
        "num_workers": 6,       # 1,  # parallelism
        "env_config": heating_controller_config.env_config_dict,
    },
}

demo_DQN_config = {
    "run": "DQN",
    "env": HeatingEnv,
    "stop": {
        "training_iteration": 200,
    },
    "checkpoint_freq": 15,
    "checkpoint_at_end": True,
    "config": {
        'model': {
            "use_lstm": ray.tune.grid_search([False]),
            "fcnet_hiddens": ray.tune.grid_search([[16, 16]]) 
            # , [32, 32], [16, 16, 16], [32, 32, 32]]),
            },
        "lr": ray.tune.grid_search([0.00002, 0.0001, 0.0005]),
        "gamma": 0.97,
        "train_batch_size": ray.tune.grid_search([256, 512, 1024]),
        "num_workers": 6,       # 1,  # parallelism
        "env_config": heating_controller_config.env_config_dict,
    },
}
        
demo_PPO_config_2 = {
    "run": "PPO",       # This algorithm yielded the best performance so far
    "env": HeatingEnv,
    "stop": {
        "training_iteration": 200,
    },
    "checkpoint_freq": 20,
    "checkpoint_at_end": True,
    "config": {
        'model': {
            "use_lstm": False,
            "fcnet_hiddens": ray.tune.grid_search([[8], [8, 8], [16], [16, 16], [32, 32]]),
            },
        "lr": ray.tune.grid_search([0.0001, 0.0005, 0.0025]),
        "train_batch_size": ray.tune.grid_search([1024, 2048, 4096]),   # 8192
        "gamma": 0.97,
        "entropy_coeff": 0.01,
        # "vf_clip_param": 10.0,
        # "grad_clip": 10,
        "num_workers": 6,       # 1,  # parallelism
        "env_config": heating_controller_config.env_config_dict,
    },
}


if __name__ == "__main__":

    ray.init()

    ray.tune.run_experiments({
        "demo_PPO_2": demo_PPO_config_2
    })
