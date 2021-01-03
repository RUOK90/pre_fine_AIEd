import argparse
import os
import random
import sys
import torch
import wandb
import numpy as np


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_run_script():
    run_script = "python"
    for e in sys.argv:
        run_script += " " + e
    return run_script


def str2bool(param):
    if param.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if param.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def print_args(parser, args):
    info = "\n[args]________________________________________________\n"
    for sub_args in parser._action_groups:
        if sub_args.title in ["positional arguments", "optional arguments"]:
            continue
        size_sub = len(sub_args._group_actions)
        info += f"├─ {sub_args.title} ({size_sub})\n"
        for i, arg in enumerate(sub_args._group_actions):
            prefix = "└─" if i == size_sub - 1 else "├─"
            info += f"│     {prefix} {arg.dest:20s}: {getattr(args, arg.dest)}\n"
    info += "└─────────────────────────────────────────────────────\n"
    print(info)


def get_arg_parser():
    parser = argparse.ArgumentParser()

    #################### Base args ####################
    base_args = parser.add_argument_group("Base args")
    base_args.add_argument("--run_script")
    base_args.add_argument("--debug_mode", type=str2bool, default=True)
    base_args.add_argument("--gpu", type=str, default="7")
    base_args.add_argument("--device", type=str)
    base_args.add_argument("--num_workers", type=int, default=4)

    #################### Logging args ####################
    logging_args = parser.add_argument_group("Logging args")
    logging_args.add_argument("--use_wandb", type=str2bool, default=False)
    logging_args.add_argument("--wandb_project", type=str, default="pre_fine_aied")
    logging_args.add_argument("--wandb_name", type=str)
    logging_args.add_argument("--wandb_tags")
    logging_args.add_argument("--use_tensorboard", type=str2bool, default=True)

    #################### Path args ####################
    path_args = parser.add_argument_group("Path args")
    path_args.add_argument(
        "--question_info_path",
        type=str,
        default="/private/datasets/LAK21_AM/question_info.csv",
    )
    path_args.add_argument(
        "--score_base_path",
        type=str,
        default="/private/datasets/LAK21_AM/score_data/14d_10q",
    )
    path_args.add_argument(
        "--pretrain_base_path",
        type=str,
        default="/private/datasets/LAK21_AM/load",
    )
    path_args.add_argument("--weight_base_path", type=str, default="/private/weights")

    #################### Train args ####################
    train_args = parser.add_argument_group("Train args")
    train_args.add_argument("--random_seed", type=int, default=1234)
    train_args.add_argument("--num_cross_folds", type=int, default=5)
    train_args.add_argument("--min_seq_size", type=int, default=11)  # +1 for cls
    train_args.add_argument("--max_seq_size", type=int, default=101)  # +1 for cls
    train_args.add_argument("--pretrain_train_batch_size", type=int, default=1024)
    train_args.add_argument("--pretrain_test_batch_size", type=int, default=2048)
    train_args.add_argument("--finetune_train_batch_size", type=int, default=256)
    train_args.add_argument("--finetune_test_batch_size", type=int, default=2048)
    train_args.add_argument(
        "--optim", type=str, choices=["scheduled", "noam"], default="scheduled"
    )
    train_args.add_argument("--lr", type=float, default=0.001)
    train_args.add_argument("--warmup_steps", type=int, default=4000)
    train_args.add_argument("--num_pretrain_epochs", type=int, default=100)
    train_args.add_argument("--num_finetune_epochs", type=int, default=100)
    train_args.add_argument("--random_mask_ratio", type=float, default=0.6)
    train_args.add_argument("--cut_point", type=float, default=0.2)
    train_args.add_argument("--aug_ratio", type=float, default=0.5)
    train_args.add_argument("--aug_sample_ratio", type=float, default=0.5)
    train_args.add_argument(
        "--aug_mode",
        type=str,
        choices=["no_aug", "aug_only", "both"],
        default="aug_only",
    )
    train_args.add_argument(
        "--gen_cate_target_sampling",
        type=str,
        choices=["none", "categorical"],
        default="categorical",
    )
    train_args.add_argument(
        "--gen_cont_target_sampling",
        type=str,
        choices=["none", "normal"],
        default="normal",
    )
    train_args.add_argument("--time_loss_lambda", type=float, default=1)
    train_args.add_argument(
        "--time_output_func",
        type=str,
        choices=["identity", "sigmoid"],
        default="sigmoid",
    )
    train_args.add_argument(
        "--time_loss", type=str, choices=["bce", "mse"], default="mse"
    )
    train_args.add_argument(
        "--finetune_output_func",
        type=str,
        choices=["mean", "cls"],
        default="mean",
    )
    train_args.add_argument(
        "--train_mode",
        type=str,
        choices=[
            "pretrain_only",
            "finetune_only",
            "finetune_only_from_pretrained_weight",
            "both",
        ],
        default="both",
    )
    train_args.add_argument("--pretrained_weight_epoch", type=int, default=2)
    train_args.add_argument(
        "--input_features",
        type=str,
        choices=[
            "qid",
            "part",
            "is_correct",
            "elapsed_time",
            "is_on_time",
            "lag_time",
        ],
        nargs="+",
        default=["qid", "part", "is_correct", "elapsed_time", "lag_time"],
    )
    train_args.add_argument(
        "--masked_features",
        type=str,
        choices=[
            "qid",
            "part",
            "is_correct",
            "elapsed_time",
            "is_on_time",
            "lag_time",
        ],
        nargs="+",
        default=["is_correct", "elapsed_time"],
    )
    train_args.add_argument(
        "--targets",
        type=str,
        choices=[
            "qid",
            "part",
            "is_correct",
            "elapsed_time",
            "is_on_time",
            "lag_time",
        ],
        nargs="+",
        default=["is_correct", "elapsed_time"],
    )
    train_args.add_argument(
        "--downstream_task",
        type=str,
        choices=["score", "dropout", "offer_acceptance"],
        default="score",
    )

    #################### Model args ####################
    model_args = parser.add_argument_group("Model args")
    model_args.add_argument(
        "--model",
        type=str,
        choices=["am", "bert", "electra", "electra-reformer"],
        default="electra",
    )
    model = parser.parse_args().model
    if model == "am":
        model_args.add_argument("--num_layers", type=int, default=2)
        model_args.add_argument("--d_model", type=int, default=256)
        model_args.add_argument("--num_heads", type=int, default=8)
        model_args.add_argument("--dropout", type=float, default=0.2)
    elif model == "electra":
        model_args.add_argument("--dis_lambda", type=int, default=1)
        model_args.add_argument("--embedding_size", type=int, default=256)
        model_args.add_argument("--hidden_size", type=int, default=256)
        model_args.add_argument(
            "--intermediate_size", type=int, default=1024
        )  # 4 * hidden_size
        model_args.add_argument("--num_hidden_layers", type=int, default=2)
        model_args.add_argument("--num_attention_heads", type=int, default=8)
        model_args.add_argument("--hidden_act", type=str, default="gelu")
        model_args.add_argument("--hidden_dropout_prob", type=float, default=0.1)
        model_args.add_argument(
            "--attention_probs_dropout_prob", type=float, default=0.1
        )
    elif model == "electra-reformer":
        model_args.add_argument("--axial_pos_embds", type=str2bool, default=True)
        model_args.add_argument(
            "--axial_pos_shape", type=int, nargs="+", default=[64, 64]
        )
        model_args.add_argument(
            "--axial_pos_embds_dim", type=int, nargs="+", default=[64, 192]
        )
        model_args.add_argument("--hidden_size", type=int, default=256)
        model_args.add_argument(
            "--hidden_act", type=str, choices=["relu", "gelu"], default="relu"
        )
        model_args.add_argument("--hidden_dropout_prob", type=float, default=0.05)
        model_args.add_argument("--feed_forward_size", type=int, default=512)
        model_args.add_argument("--attention_head_size", type=int, default=64)
        model_args.add_argument(
            "--attn_layers",
            type=str,
            choices=["local", "lsh"],
            nargs="+",
            default=["local", "lsh", "local", "lsh", "local", "lsh"],
        )
        model_args.add_argument("--num_attention_heads", type=int, default=12)
        model_args.add_argument("--local_attn_chunk_length", type=int, default=64)
        model_args.add_argument(
            "--local_attention_probs_dropout_prob", type=float, default=0.05
        )
        model_args.add_argument("--local_num_chunks_before", type=int, default=1)
        model_args.add_argument("--local_num_chunks_after", type=int, default=0)
        model_args.add_argument("--lsh_attn_chunk_length", type=int, default=64)
        model_args.add_argument(
            "--lsh_attention_probs_dropout_prob", type=float, default=0
        )
        model_args.add_argument("--lsh_num_chunks_before", type=int, default=1)
        model_args.add_argument("--lsh_num_chunks_after", type=int, default=0)
        model_args.add_argument("--num_hashes", type=int, default=1)
        model_args.add_argument("--num_buckets", default=None)
        model_args.add_argument("--is_decoder", type=str2bool, default=False)
        model_args.add_argument("--use_cache", type=str2bool, default=False)

    return parser


def get_args():
    parser = get_arg_parser()
    args = parser.parse_args()
    args.run_script = get_run_script()

    # name
    # input_masked_target
    args.wandb_name = ""
    # input
    if "is_correct" in args.input_features:
        args.wandb_name += "ic-"
    if "is_on_time" in args.input_features:
        args.wandb_name += "iot-"
    if "elapsed_time" in args.input_features:
        args.wandb_name += "et-"
    if "lag_time" in args.input_features:
        args.wandb_name += "lt-"
    args.wandb_name = args.wandb_name.rstrip("-") + "_"
    # target
    if "is_correct" in args.targets:
        args.wandb_name += "ic-"
    if "is_on_time" in args.targets:
        args.wandb_name += "iot-"
    if "elapsed_time" in args.targets:
        args.wandb_name += "et-"
    if "lag_time" in args.targets:
        args.wandb_name += "lt-"
    args.wandb_name = args.wandb_name.rstrip("-")
    args.wandb_name += (
        f"_{args.gen_cate_target_sampling}_{args.gen_cont_target_sampling}"
    )
    if args.gen_cont_target_sampling == "none":
        args.wandb_name += f"_{args.time_output_func}_{args.time_loss}"

    # parse tags
    args.wandb_tags = (
        args.wandb_tags.split(",") if args.wandb_tags is not None else ["test"]
    )
    args.wandb_tags.append(args.wandb_name)

    # random seed
    set_random_seed(args.random_seed)

    # debug
    if args.debug_mode:
        args.num_workers = 0
        args.num_cross_folds = 1
        args.score_base_path = "/private/datasets/LAK21_AM/score_data_debug/14d_10q"
        args.pretrain_base_path = "/private/datasets/LAK21_AM/load_debug"
        args.pretrain_train_batch_size = 4
        args.pretrain_test_batch_size = 4
        args.finetune_train_batch_size = 4
        args.finetune_test_batch_size = 4
        args.num_pretrain_epochs = 5
        args.num_finetune_epochs = 5
        args.wandb_name = "debug"

    # wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=args.wandb_tags,
            config=args,
        )

    # parse gpus
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        assert torch.cuda.is_available()
        args.device = "cuda"
    else:
        args.device = "cpu"

    # get weight path
    args.weight_path = f"{args.weight_base_path}/{args.model}/{args.wandb_name}"
    os.makedirs(args.weight_path, exist_ok=True)

    # get all features
    args.all_features = args.input_features + args.targets
    args.all_features.sort()

    # time output_func and loss sanity check
    if args.time_loss == "bce":
        assert args.time_output_func == "sigmoid"

    # electra sanity check
    if args.model == "electra":
        assert set(args.masked_features) == set(args.targets)

    return args, parser


ARGS, parser = get_args()
print_args(parser, ARGS)


class Const:
    UNKNOWN_PART = 8
    MAX_ELAPSED_TIME_IN_S = 300
    DEFAULT_TIME_LIMIT_IN_MS = 43000
    MAX_LAG_TIME_IN_S = 86400

    PAD_VAL = 0
    TRUE_VAL = 1
    FALSE_VAL = 2

    CATE_VARS = ["qid", "part", "is_correct", "is_on_time"]
    CONT_VARS = ["elapsed_time", "lag_time"]

    FEATURE_SIZE = {
        "qid": None,  # initialized in get_q_info_dic
        "part": 8,  # 8 for unknown
        "is_correct": 2,
        "is_on_time": 2,
        "elapsed_time": None,  # continuous var
        "lag_time": None,  # continuous var
    }

    CLS_VAL = {
        "qid": None,  # initialized in get_q_info_dic
        "part": FEATURE_SIZE["part"] + 1,
        "is_correct": FEATURE_SIZE["is_correct"] + 1,
        "is_on_time": FEATURE_SIZE["is_on_time"] + 1,
        "elapsed_time": 0,
        "lag_time": 0,
    }

    MASK_VAL = {
        "qid": None,  # initialized in get_q_info_dic
        "part": FEATURE_SIZE["part"] + 2,
        "is_correct": FEATURE_SIZE["is_correct"] + 2,
        "is_on_time": FEATURE_SIZE["is_on_time"] + 2,
        "elapsed_time": -1,
        "lag_time": -1,
    }

    SCORE_SCALING_FACTOR = 495
