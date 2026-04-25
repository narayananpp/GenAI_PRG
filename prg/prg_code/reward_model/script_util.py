"""
Helpers for constructing reward models from flat argument dictionaries.
Mirrors guided_diffusion/script_util.py.
"""

from .model import create_reward_model


def reward_model_defaults():
    """
    Defaults for the reward model architecture.
    These are added to the argparser so every flag is exposed from the CLI.
    """
    return dict(
        in_channels=263,
        model_channels=256,
        num_res_blocks=4,
        num_heads=4,
        ffn_mult=2,
        dropout=0.1,
        max_seq_len=196,
    )


def create_reward_model_from_args(args):
    """
    Instantiate a reward model from a parsed argparse Namespace.

    :param args: argparse.Namespace containing reward_model_defaults() keys.
    :return: FeasibilityRewardModel.
    """
    return create_reward_model(**args_to_dict(args, reward_model_defaults().keys()))


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys if hasattr(args, k)}


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"boolean value expected, got {v}")
