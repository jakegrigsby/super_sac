import gin
import os

import super_sac
from super_sac import nets
from super_sac.augmentations import AugmentationSequence, Drqv2Aug


class AtariEncoder(nets.Encoder):
    def __init__(self, img_shape, emb_dim=128):
        super().__init__()
        self._emb_dim = emb_dim
        self.cnn = nets.cnns.SmallPixelEncoder(img_shape, emb_dim)

    @property
    def embedding_dim(self):
        return self._emb_dim

    def forward(self, obs_dict):
        return self.cnn(obs_dict["obs"])


def train_atari(args):
    gin.parse_config_file(args.config)

    def make_env():
        return super_sac.wrappers.load_atari(args.game, frame_skip=4)

    train_env = super_sac.wrappers.Uint8Wrapper(
        super_sac.wrappers.ParallelActors(make_env, args.parallel_actors)
    )
    test_env = super_sac.wrappers.Uint8Wrapper(make_env())

    # create agent
    img_shape = train_env.observation_space.shape
    agent = super_sac.Agent(
        act_space_size=train_env.action_space.n,
        encoder=AtariEncoder(img_shape, emb_dim=128),
    )

    buffer = super_sac.replay.PrioritizedReplayBuffer(size=1_000_000)

    # run training
    super_sac.super_sac(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        name=args.name,
        augmenter=AugmentationSequence([Drqv2Aug(128)]),
        logging_method="wandb",
        wandb_entity=os.getenv("SSAC_WANDB_ACCOUNT"),
        wandb_project=os.getenv("SSAC_WANDB_PROJECT"),
        base_save_path=os.getenv("SSAC_SAVE"),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of logging dir", required=True)
    parser.add_argument("--game", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--parallel_actors", type=int, default=1)
    args = parser.parse_args()
    train_atari(args)
