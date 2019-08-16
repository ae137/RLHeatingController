from heating_control import HeatingEnv, HeatingEnvCreator

import matplotlib.pyplot as plt
import numpy as np
import gym
import ray.rllib.agents.ppo as ppo
import ray

import temperature_simulator as temp_sim

ray.tune.registry.register_env('HeatingEnv-v0', HeatingEnvCreator)

t_init = 0
To_min = 5 # -5
To_max = 15 # 10
T_target = 21

config = {
    'model': {
        # "use_lstm": True,
        "fcnet_hiddens": [32, 32],
        },
    "lr": 0.0005,
    "train_batch_size": 4096,
    "num_workers": 6, # 1,  # parallelism
    "env_config": {
        "h": 0.15,
        "l": 0.025,
        "temp_diff_penalty": 1,
        'horizon': 400,
        "out_temp_sim": temp_sim.EnvironmentSimulator(t_init, To_min, To_max),
        "target_temp": temp_sim.TargetTemperature(t_init, T_target)
        }
    }

base_path = '/home/andreas/ray_results/demo'
trial_path = 'PPO_HeatingEnv_0_lr=0.0005_2019-08-16_11-14-05x5eybj01'
checkpoint_num = 100
    
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

obs = np.array(env.init_state)



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
