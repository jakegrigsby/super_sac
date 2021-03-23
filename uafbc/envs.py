import gym


class SimpleGymWrapper(gym.wrappers.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return {"obs": obs}
