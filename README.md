# Super SAC

This repository contains the code for a PyTorch RL agent that is designed to be a compilation of advanced Off-Policy Actor-Critic variants. It includes elements of [SAC](https://arxiv.org/abs/1812.05905), [SAC-Discrete](https://arxiv.org/abs/1910.07207), [DrQ](https://arxiv.org/abs/2004.13649), [DrAC](https://arxiv.org/abs/2006.12862), [RAD](https://arxiv.org/abs/2004.14990), [SAC+AE](https://arxiv.org/abs/1910.01741), [REDQ](https://arxiv.org/abs/2101.05982), [CRR](https://arxiv.org/abs/2006.15134), [AWAC](https://arxiv.org/abs/2006.09359), and [SUNRISE](https://arxiv.org/abs/2007.04938).

**This is an active work in progress and I add breaking features often as I need them in my research. If you have any suggestions or questions please feel free to open an issue.**

### Main Features:
- Dictionary-based observations. This allows for multimodal/goal-oriented states and support for a much wider range of environments. Basic gym envs can be wrapped to return a dictionary with one key.
- Process high-dimensional states with a dedicated `Encoder` module designed for each environment. Actor and Critic networks are basic MLPs that interpret the output of the encoder instead of the observation dictionary. Basic gym envs can use an identity encoder to recover the normal setup. The encoder is only trained with gradients from the critic loss function, as in e.g. [DrQ](https://arxiv.org/abs/2004.13649), [SAC+AE](https://arxiv.org/abs/1910.01741).
- MaxEnt RL with automatic entropy tuning, as in [SAC](https://arxiv.org/abs/1812.05905).
- Continuous and discrete action spaces, as in [SAC-Discrete](https://arxiv.org/abs/1910.07207).
- Warmup learning with behavioral cloning from an offline dataset.
- A hybrid version of [CRR](https://arxiv.org/abs/2006.15134) and [AWAC](https://arxiv.org/abs/2006.09359) for Offline RL and Offline pre-training. I studied this specific implementation with prioritized replay in [my undergrad thesis](https://csdmp.github.io/docs/grigsby2021.pdf).
- Data Augmentation as in [DrQ](https://arxiv.org/abs/2004.13649), [RAD](https://arxiv.org/abs/2004.14990), with regularization as in [DrAC](https://arxiv.org/abs/2006.12862).
- Ensemble of critic networks with the generalized clipped-double-q-trick from [REDQ](https://arxiv.org/abs/2101.05982).
- Critic-uncertainty exploration incentive as in [SUNRISE](https://arxiv.org/abs/2007.04938).
- Ensemble of actor networks, loosely based on [SUNRISE](https://arxiv.org/abs/2007.04938).
- Weighted critic updates based on ensemble uncertainty to prevent error propagation, loosely based on [SUNRISE](https://arxiv.org/abs/2007.04938).
- Parallel environment collection (with a lot of help from [stable_baselines3](https://github.com/DLR-RM/stable-baselines3)).
- Tensorboard logging.

### Examples:
At some point I will write real documentation. For now see the `examples/` folder. There are examples training scripts for:
- Basic continuous control tasks in [OpenAI Gym](https://gym.openai.com) / [DeepMind Control Suite](https://arxiv.org/abs/1801.00690) / [PyBullet](https://github.com/bulletphysics/bullet3).
- Image-based discrete-action games from Atari, [MinAtar](https://github.com/kenjyoung/MinAtar).
- Offline RL in Atari, [D4RL](https://github.com/rail-berkeley/d4rl).
- Image-based continuous control from [DMCR](https://arxiv.org/abs/2010.06740).

(Please Note: examples that rely on MuJoCo (some gym tasks, D4RL, DMCR) are the least tested because my student licenses have expired since I implemented them...)



