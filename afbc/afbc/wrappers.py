import gym
import numpy as np


class NormActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._true_action_space = env.action_space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32,
        )

    def action(self, action):
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self.action_space.high - self.action_space.low
        action = (action - self.action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        return action


class DiscreteActionWrapper(gym.ActionWrapper):
    def action(self, action):
        if len(action.shape) > 0:
            action = action[0]
        action = int(action)
        return action


class SimpleGymWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return {"obs": obs.astype(np.float32)}


class KeepKeysWrapper(gym.ObservationWrapper):
    def __init__(self, env, keys):
        super().__init__(env)
        self.keys = keys

    def observation(self, obs):
        return {key: obs[key] for key in self.keys}
