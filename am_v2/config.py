"""
CLI argument parsing and config file parsing.
"""

import argparse
import os
import random
import sys

import torch

from am_v2 import util


def get_run_script():
    """
    Returns the command-line command used to run job. For logging.
    """
    run_script = "python" + " " + " ".join(sys.argv)
    return run_script


def str2bool(param):
    """
    Convert string type params to boolean
    TODO: Replace str2bool with the action store_true.
    Then
    train_args.add_argument("--is_augmented", type=str2bool)
    will become
    train_args.add_argument("--is_augmented", action='store_true')
    This way, unspecified flags will resolve to None and all boolean
    flags that are given (e.g. --is_augmented) will resolve
    automatically to True.

    For more information see:
    https://docs.python.org/3/library/argparse.html#action
    """
    if param.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if param.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def check_params(params):
    """
    Sanity check on arguments.
    """
    if params.tags is None or params.num_epochs < 1:
        assert False


def _read_config(fpath):
    """
    Parse configuration file.
    """
    print(f"reading config file: {fpath}")
    output = []
    with open(fpath, "r") as lines:
        for line in lines:
            split = line.split(" ")
            if len(split) < 2:
                print(f"The line {line} is not a valid configuration")
                raise ValueError
            split[-1] = split[-1][:-1]  # strip newline
            output.extend(split)
    return output


def _get_arg_parser():
    """
    Returns argparse parser with arguments added.
    """
    parser = argparse.ArgumentParser()

    base_args = parser.add_argument_group("Base args")
    train_args = parser.add_argument_group("Train args")

    base_args.add_argument(
        "--config", type=_read_config, default="am_v2/configs/base_config.txt"
    )
    base_args.add_argument("--run_script")
    base_args.add_argument("--debug_mode", type=str2bool)
    base_args.add_argument("--save_path")
    base_args.add_argument("--tags")
    base_args.add_argument("--project", type=str)
    base_args.add_argument("--name", type=str)
    base_args.add_argument("--device", type=str)
    base_args.add_argument("--data_file", type=str)
    base_args.add_argument("--use_data_tree", type=str2bool)
    base_args.add_argument("--gpu", type=str)
    base_args.add_argument("--num_workers", type=int)
    base_args.add_argument("--base_path", type=str)
    base_args.add_argument("--machine_name", type=str)
    train_args.add_argument("--num_cross_folds", type=int)
    train_args.add_argument("--cut_point", type=float)
    train_args.add_argument("--random_seed", type=int)
    train_args.add_argument("--num_epochs", type=int)
    train_args.add_argument("--train_batch", type=int)
    train_args.add_argument("--test_batch", type=int)
    train_args.add_argument("--lr", type=float)
    train_args.add_argument("--cf_mode", type=str)
    train_args.add_argument("--seq_size", type=int)
    train_args.add_argument("--sample_rate", type=float)
    train_args.add_argument("--is_augmented", type=str2bool)
    train_args.add_argument("--use_pretrain", type=str2bool)
    train_args.add_argument("--use_old_split", type=str2bool)
    train_args.add_argument("--use_new_score_data", type=str2bool)
    train_args.add_argument("--use_score_val", type=str2bool)
    train_args.add_argument("--use_session", type=str2bool)
    train_args.add_argument("--is_freeze", type=str2bool)
    train_args.add_argument("--warmup_steps", type=int)
    train_args.add_argument("--weighted_loss", type=str2bool)
    train_args.add_argument("--num_layer", type=int)
    train_args.add_argument("--d_model", type=int)
    train_args.add_argument("--num_heads", type=int)
    train_args.add_argument("--random_rate", type=float)
    train_args.add_argument("--dropout", type=float)
    train_args.add_argument("--weight_dir", type=str)
    train_args.add_argument("--lambda1", type=float)
    train_args.add_argument("--lambda2", type=float)
    train_args.add_argument("--use_buggy_timeliness", type=str2bool)
    train_args.add_argument("--pretrain_root_path", type=str)  # csv ex: 2,5,7
    train_args.add_argument("--pretrain_number", type=int)  # epoch num
    train_args.add_argument("--pretrain_name", type=str)
    train_args.add_argument("--pretrain_epoch_start_index", type=int)
    train_args.add_argument("--pretrain_epoch_end_index", type=int)
    train_args.add_argument("--min_seq", type=int)
    train_args.add_argument("--is_warm_up", type=str2bool)
    train_args.add_argument("--mask_position", type=str)  # head | tail | random
    train_args.add_argument("--is_pretrain", type=str2bool)
    train_args.add_argument("--mask_random_ratio", type=float)
    train_args.add_argument("--mask_rate", type=float)
    train_args.add_argument("--is_feature_based", type=str2bool)
    train_args.add_argument("--use_only_is_on_time", type=str2bool)
    train_args.add_argument("--freeze_embedding", type=str2bool)
    train_args.add_argument("--freeze_encoder_block", type=str2bool)
    train_args.add_argument("--is_test_with_pretrain", type=str2bool)
    train_args.add_argument("--max_elapsed_time", type=int)
    train_args.add_argument("--train_separate", type=str2bool)
    train_args.add_argument("--elapsed_input_form", type=str)
    train_args.add_argument("--time_output_form", type=str)
    train_args.add_argument("--load_question_embed", type=str)
    train_args.add_argument(
        "--pretrain_task",
        type=str,
        choices=[
            "is_on_time",
            "is_correct",
            "is_long_option",
            "startime",
            "session_order",
            "elapsed_time",
            "lag_time",
        ],
        nargs="+",
    )
    train_args.add_argument(
        "--time_format",
        type=str,
        choices=["is_on_time", "discrete_elapsed_time", "continuous_elapsed_time"],
    )
    train_args.add_argument(
        "--add_pre_train_task",
        type=str,
        choices=["None", "is_select_long_option", "is_confused", "is_guessed"],
    )
    train_args.add_argument("--score_loss", type=str, choices=["l1", "l2", "bce"])
    train_args.add_argument(
        "--score_last_activation", type=str, choices=["none", "sigmoid", "tanh"]
    )
    train_args.add_argument("--score_scaling_factor", type=int)
    train_args.add_argument("--log_score_outputs", type=str2bool)

    return parser


def _get_args():
    """
    Returns: args, an argparse Namespace
    """
    parser = _get_arg_parser()

    def _print_args(params, is_write):

        info = "\n[args]________________________________________________\n"
        for sub_args in parser._action_groups:
            if sub_args.title in ["positional arguments", "optional arguments"]:
                continue
            size_sub = len(sub_args._group_actions)
            info += f"├─ {sub_args.title} ({size_sub})\n"
            for i, arg in enumerate(sub_args._group_actions):
                prefix = "└─" if i == size_sub - 1 else "├─"
                info += f"│     {prefix} {arg.dest:20s}: {getattr(params, arg.dest)}\n"
        info += "└─────────────────────────────────────────────────────\n"
        print(info)

        if is_write:
            util.write(params.save_path, info, mode="w")

    _args = parser.parse_args()  # Set default args to access config
    _args = parser.parse_args(_args.config, _args)  # Set config args
    # Override with args from CLI, first None defaults to sys.argv
    _args = parser.parse_args(None, _args)

    _args.run_script = get_run_script()

    # check_args
    if _args.debug_mode is False:
        check_params(_args)

    # tag&save
    if _args.tags is not None:
        _args.tags = _args.tags.split(",")
    else:
        _args.tags = ["test"]

    # random_seed
    # TODO: Move this to main.py
    torch.manual_seed(_args.random_seed)
    torch.cuda.manual_seed_all(_args.random_seed)
    random.seed(_args.random_seed)

    if torch.cuda.is_available():
        _args.device = "cuda"
        if _args.gpu is not None:
            _args.gpu = [int(e) for e in _args.gpu.split(",")]
            torch.cuda.set_device(_args.gpu[0])

    _args.weight_path = f"{_args.weight_dir}/{_args.name}"
    _args.save_path = f"log/{_args.name}.log"
    os.makedirs(_args.weight_path, exist_ok=True)

    _print_args(_args, False)

    return _args


ARGS = _get_args()
