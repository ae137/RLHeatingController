import ray.rllib.policy.policy as policy


class BaselinePolicy(policy.Policy):
    """Example of a custom policy written from scratch.

    You might find it more convenient to use the `build_tf_policy` and
    `build_torch_policy` helpers instead for a real policy, which are
    described in the next sections.
    """

    def __init__(self, observation_space, action_space, config):
        policy.Policy.__init__(self, observation_space, action_space, config)
        # example parameter

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return action batch, RNN states, extra values to include in batch
        return [self.compute_action(obs) for obs in obs_batch], [], {}

    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}  # return stats


class RandomPolicy(BaselinePolicy):
    def compute_action(self, obs):
        return self.action_space.sample()


class HeatWhenTooColdPolicy(BaselinePolicy):
    def compute_action(self, obs):
        return int(obs[3] < obs[1])
