'''
Documentation, License etc.

@package heating_control
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
import math
import random
from gym.spaces import Discrete, Box

import ray
from ray.tune import run_experiments, grid_search

import temperature_simulator as temp_sim

random.seed(0)

class HeatingEnv(gym.Env):
    episode_counter = 0
    
    action_space = Discrete(2)
    observation_space = Box(-30, 50, shape=(2,), dtype=np.float32)

    
    def __init__(self, config):
        self.h = config["h"]
        self.l = config["l"]
        self.horizon = config['horizon']
        self.temp_diff_penalty = config["temp_diff_penalty"]
        
        self.out_temp_sim = config["out_temp_sim"]
        self.target_temp = config["target_temp"]
        self.time = self.out_temp_sim.getInitTime()
        
        self.init_state = (20, self.out_temp_sim.getOutTemp(self.time))
        self.state = self.init_state
    
    def reset(self):
        self.time = self.out_temp_sim.getInitTime()
        # As initial temperature in the room, we set 20 °C for now. This should be more random
        self.state = (self.init_state[0] + 0 * random.randint(-10, 10), self.init_state[1])
        HeatingEnv.episode_counter += 1
        
        return np.array(self.state)

    def step(self, action):
        # action = self.action_space.sample()       # Random agent
        
        if (self.time > self.horizon):     # Specify conditions for end of episode
            return (self.reset(), 0., True, {})
        
        # Compute new time
        self.time += 1
        # Store old Temperatures
        # Ti_old, To_old = self.state.get()
        Ti_old, To_old = self.state[0], self.state[1]
        
        Ti_new = Ti_old + math.sqrt(max(Ti_old - To_old, 1)) \
            * self.h * action + (To_old - Ti_old) * self.l
        To_new = self.out_temp_sim.getOutTemp(self.time)
        
        self.state = (Ti_new, To_new)
        
        reward = -self.temp_diff_penalty \
            * abs(self.target_temp.getTargetTemp(self.time) - Ti_new)**2
        
        return (np.array(self.state), reward, False, {})
    
def HeatingEnvCreator(env_config):
    return HeatingEnv(env_config)

if __name__ == "__main__":    
    t_init = 0
    To_min = 5 # -5
    To_max = 15 # 10
    T_target = 21
    
    ray.init()
    # ModelCatalog.register_custom_model("my_model", CustomModel)
    run_experiments({
        "demo": {
            "run": "PPO",       # This algorithm seems to yield the best performance
            # "run": "DQN",
            # "run": "A3C",
            "env": HeatingEnv,
            "stop": {
                "training_iteration": 100, # 10
            },
            "checkpoint_freq": 9,
            "checkpoint_at_end": True,
            "config": {
                # "model": {
                #     "custom_model": "my_model",
                # },
                'model': {
                    # "use_lstm": True,
                    "fcnet_hiddens": [32, 32],
                    },
                "lr": grid_search([0.0005]), # [1e-3, 1e-4, 1e-5]
                "train_batch_size": 4096,
                "num_workers": 6, # 1,  # parallelism
                "env_config": {
                    "h": 0.15,
                    "l": 0.025,
                    "temp_diff_penalty": 1,
                    'horizon': 400,
                    "out_temp_sim": temp_sim.EnvironmentSimulator(t_init, To_min, To_max),
                    "target_temp": temp_sim.TargetTemperature(t_init, T_target)
                    },
            },
        },
    })
