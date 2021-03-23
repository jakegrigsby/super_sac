import argparse


def add_args(parser):
    parser.add_argument(
        "--num_steps_offline",
        type=int,
        default=10_000,
        help="Number of steps of offline learning",
    )
    parser.add_argument(
        "--num_steps_online",
        type=int,
        default=0,
        help="Number of steps of online learning",
    )
    parser.add_argument(
        "--transitions_per_online_step",
        type=int,
        default=1,
        help="env transitions per training step. Defaults to 1, but will need to \
        be set higher for repaly ratios < 1",
    )
    parser.add_argument(
        "--max_episode_steps", type=int, default=1000, help="maximum steps per episode",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="training batch size"
    )
    parser.add_argument(
        "--tau", type=float, default=0.005, help="for model parameter % update"
    )
    parser.add_argument(
        "--actor_lr", type=float, default=1e-4, help="actor learning rate"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=1e-4, help="critic learning rate"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="gamma, the discount factor"
    )
    parser.add_argument(
        "--buffer_size", type=int, default=1_000_000, help="replay buffer size"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1,
        help="how often to test the agent without exploration (in episodes)",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=100,
        help="how many episodes to run for when testing",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="flag to enable env rendering during training",
    )
    parser.add_argument(
        "--actor_clip",
        type=float,
        default=None,
        help="gradient clipping for actor updates",
    )
    parser.add_argument(
        "--critic_clip",
        type=float,
        default=None,
        help="gradient clipping for critic updates",
    )
    parser.add_argument(
        "--name", type=str, default="efr_mc1d", help="dir name for saves"
    )
    parser.add_argument(
        "--actor_l2",
        type=float,
        default=0.0,
        help="L2 regularization coeff for actor network",
    )
    parser.add_argument(
        "--critic_l2",
        type=float,
        default=0.0,
        help="L2 regularization coeff for critic network",
    )
    parser.add_argument(
        "--target_delay",
        type=int,
        default=2,
        help="How many steps to go between target network updates",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="How many steps to go between saving the agent params to disk",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="verbosity > 0 displays a progress bar during training",
    )
    parser.add_argument(
        "--actor_updates_per_step",
        type=int,
        default=100,
        help="how many gradient updates to make per training step",
    )
    parser.add_argument(
        "--critic_updates_per_step",
        type=int,
        default=250,
        help="how many gradient updates to make per training step",
    )
    parser.add_argument(
        "--prioritized_replay",
        action="store_true",
        help="flag that enables use of prioritized experience replay",
    )
    parser.add_argument(
        "--skip_save_to_disk",
        action="store_true",
        help="flag to skip saving agent params to disk during training",
    )
    parser.add_argument(
        "--skip_log_to_disk",
        action="store_true",
        help="flag to skip saving agent performance logs to disk during training",
    )
    parser.add_argument(
        "--log_std_low",
        type=float,
        default=-10,
        help="Lower bound for log std of action distribution.",
    )
    parser.add_argument(
        "--log_std_high",
        type=float,
        default=2,
        help="Upper bound for log std of action distribution.",
    )
    parser.add_argument(
        "--adv_method",
        type=str,
        default="mean",
        help="Approach for estimating the advantage function. Choices include {'max', 'mean'}.",
    )
    parser.add_argument(
        "--adv_ensembling",
        type=str,
        default="min",
        help="Approach for estimating the advantage function. Choices include {'min', 'mean'}.",
    )
    parser.add_argument(
        "--critic_ensemble_size",
        type=int,
        default=5,
        help="how many critics in the ensemble",
    )
    parser.add_argument(
        "--no_filter", action="store_true",
    )
    parser.add_argument("--art", action="store_true")
    parser.add_argument(
        "--weight_type",
        type=str,
        choices=["sunrise", "custom", "softmax"],
        default="softmax",
    )
    parser.add_argument(
        "--actor_type", type=str, choices=["pos", "both"], default="pos"
    )
    parser.add_argument("--pop", action="store_true")
    parser.add_argument("--bc_warmup_steps", type=int, default=25_000)
