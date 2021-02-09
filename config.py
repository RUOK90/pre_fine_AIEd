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
    base_args.add_argument("--debug_mode", type=str2bool, default=False)
    base_args.add_argument("--gpu", type=str, default="7")
    base_args.add_argument("--device", type=str)
    base_args.add_argument("--num_workers", type=int, default=4)

    #################### Logging args ####################
    logging_args = parser.add_argument_group("Logging args")
    logging_args.add_argument("--use_wandb", type=str2bool, default=True)
    logging_args.add_argument("--use_finetune_wandb", type=str2bool, default=False)
    logging_args.add_argument("--wandb_project", type=str, default="pre_fine_aied")
    logging_args.add_argument(
        "--finetune_wandb_project", type=str, default="pre_fine_aied_finetune"
    )
    logging_args.add_argument("--wandb_name", type=str)
    logging_args.add_argument("--wandb_tags", default="none")

    #################### Path args ####################
    path_args = parser.add_argument_group("Path args")
    path_args.add_argument(
        "--question_info_path",
        type=str,
        default="/private/datasets/magneto_2021-01-27/questions.hdf5",
    )
    path_args.add_argument(
        "--interaction_base_path",
        type=str,
        default="/private/datasets/magneto_2021-01-27/user_interactions_wo_lecture.pkl",
    )
    path_args.add_argument(
        "--pretrain_base_path",
        type=str,
        default="/private/datasets/magneto_2021-01-27/user_interaction_windows",
    )
    path_args.add_argument(
        "--score_base_path",
        type=str,
        default="/private/datasets/magneto_2021-01-27/user_score_idxs.pkl",
    )
    path_args.add_argument("--weight_base_path", type=str, default="/private/weights")

    #################### Train args ####################
    train_args = parser.add_argument_group("Train args")
    train_args.add_argument("--random_seed", type=int, default=1234)
    train_args.add_argument("--num_cross_folds", type=int, default=5)
    train_args.add_argument("--min_seq_size", type=int, default=11)  # +1 for cls

    # train_args.add_argument("--max_seq_size", type=int, default=101)  # +1 for cls
    # train_args.add_argument("--pretrain_train_batch_size", type=int, default=1024)
    # train_args.add_argument("--pretrain_test_batch_size", type=int, default=2048)
    # train_args.add_argument("--pretrain_max_num_evals", type=int, default=100)
    # train_args.add_argument("--pretrain_update_steps", type=int, default=350)

    train_args.add_argument("--max_seq_size", type=int, default=8192)  # +1 for cls
    train_args.add_argument("--pretrain_train_batch_size", type=int, default=None)
    train_args.add_argument("--pretrain_test_batch_size", type=int, default=None)
    train_args.add_argument("--pretrain_max_num_evals", type=int, default=None)
    train_args.add_argument("--pretrain_update_steps", type=int, default=None)

    train_args.add_argument("--finetune_train_batch_size", type=int, default=None)
    train_args.add_argument("--finetune_test_batch_size", type=int, default=None)
    train_args.add_argument("--finetune_max_num_evals", type=int, default=None)
    train_args.add_argument("--finetune_update_steps", type=int, default=None)
    train_args.add_argument("--finetune_patience", type=int, default=None)

    train_args.add_argument(
        "--optim", type=str, choices=["scheduled", "noam"], default="noam"
    )
    train_args.add_argument("--lr", type=float, default=0.001)
    train_args.add_argument("--warmup_steps", type=int, default=4000)
    train_args.add_argument("--random_mask_ratio", type=float, default=0.6)
    train_args.add_argument("--dis_lambda", type=int, default=1)
    train_args.add_argument("--aug_ratio", type=float, default=0.5)
    train_args.add_argument("--aug_sample_ratio", type=float, default=0.5)
    train_args.add_argument(
        "--aug_mode",
        type=str,
        choices=["no_aug", "aug_only", "both"],
        default="no_aug",
    )
    train_args.add_argument(
        "--gen_cate_target_sampling",
        type=str,
        choices=["none", "categorical"],
        default="categorical",
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
        "--score_loss", type=str, choices=["bce", "mse"], default="bce"
    )
    train_args.add_argument(
        "--finetune_output_func",
        type=str,
        choices=["mean", "cls"],
        default="cls",
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
        default="pretrain_only",
    )
    train_args.add_argument("--pretrained_weight_n_eval", type=int, default=2)
    train_args.add_argument(
        "--input_features",
        type=str,
        choices=[
            "qid",
            "part",
            "choice",
            "is_correct",
            "elapsed_time",
            "is_on_time",
            "lag_time",
            "exp_time",
        ],
        nargs="+",
        default=[
            "qid",
            "part",
            "choice",
            "is_correct",
            "elapsed_time",
            "lag_time",
            "exp_time",
        ],
    )
    train_args.add_argument(
        "--masked_features",
        type=str,
        choices=[
            "choice",
            "is_correct",
            "elapsed_time",
            "is_on_time",
            "lag_time",
            "exp_time",
        ],
        nargs="+",
        default=[
            "choice",
            "is_correct",
            "elapsed_time",
            "lag_time",
            "exp_time",
        ],
    )
    train_args.add_argument(
        "--targets",
        type=str,
        choices=[
            "choice",
            "is_correct",
            "elapsed_time",
            "is_on_time",
            "lag_time",
            "exp_time",
        ],
        nargs="+",
        default=[
            "choice",
            "is_correct",
            "elapsed_time",
            "lag_time",
            "exp_time",
        ],
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
        choices=["am", "bert", "electra", "electra-reformer", "electra-performer"],
        default="electra-performer",
    )
    model_args.add_argument("--axial_pos_embds", type=str2bool, default=True)
    model_args.add_argument("--axial_pos_shape", type=int, nargs="+", default=[32, 32])
    model_args.add_argument(
        "--axial_pos_embds_dim", type=int, nargs="+", default=[64, 192]
    )
    model_args.add_argument("--embedding_size", type=int, default=256)
    model_args.add_argument("--hidden_size", type=int, default=256)
    model_args.add_argument("--feedforward_mult", type=int, default=4)
    model_args.add_argument("--num_hidden_layers", type=int, default=4)
    model_args.add_argument("--num_attn_heads", type=int, default=8)
    model_args.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    model_args.add_argument("--attn_probs_dropout_prob", type=float, default=0.1)
    model_args.add_argument("--num_random_features", type=int, default=256)
    model_args.add_argument("--feature_redraw_interval", type=int, default=1000)
    model_args.add_argument("--use_generalized_attn", type=str2bool, default=True)
    model_args.add_argument("--use_scale_norm", type=str2bool, default=True)
    model_args.add_argument("--use_rezero", type=str2bool, default=False)
    model_args.add_argument("--use_glu", type=str2bool, default=False)
    model_args.add_argument("--causal", type=str2bool, default=False)
    model_args.add_argument("--cross_attend", type=str2bool, default=False)

    # # if model == "am":
    # model_args.add_argument("--num_layers", type=int, default=2)
    # model_args.add_argument("--d_model", type=int, default=256)
    # model_args.add_argument("--num_heads", type=int, default=8)
    # model_args.add_argument("--dropout", type=float, default=0.2)
    # elif model == "electra":
    # model_args.add_argument("--embedding_size", type=int, default=256)
    # model_args.add_argument("--hidden_size", type=int, default=256)
    # model_args.add_argument(
    #     "--intermediate_size", type=int, default=1024
    # )  # 4 * hidden_size
    # model_args.add_argument("--num_hidden_layers", type=int, default=2)
    # model_args.add_argument("--num_attention_heads", type=int, default=8)
    # model_args.add_argument("--hidden_act", type=str, default="gelu")
    # model_args.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    # model_args.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    # # elif model == "electra-reformer":
    # model_args.add_argument("--hidden_size", type=int, default=256)
    # model_args.add_argument(
    #     "--hidden_act", type=str, choices=["relu", "gelu"], default="relu"
    # )
    # model_args.add_argument("--hidden_dropout_prob", type=float, default=0.05)
    # model_args.add_argument("--feed_forward_size", type=int, default=1024)
    # model_args.add_argument("--attention_head_size", type=int, default=64)
    # model_args.add_argument(
    #     "--attn_layers",
    #     type=str,
    #     choices=["local", "lsh"],
    #     nargs="+",
    #     default=["lsh", "lsh", "lsh", "lsh"],
    # )
    # model_args.add_argument("--num_attention_heads", type=int, default=8)
    # model_args.add_argument("--local_attn_chunk_length", type=int, default=128)
    # model_args.add_argument(
    #     "--local_attention_probs_dropout_prob", type=float, default=0.05
    # )
    # model_args.add_argument("--local_num_chunks_before", type=int, default=1)
    # model_args.add_argument("--local_num_chunks_after", type=int, default=0)
    # model_args.add_argument("--lsh_attn_chunk_length", type=int, default=128)
    # model_args.add_argument("--lsh_attention_probs_dropout_prob", type=float, default=0)
    # model_args.add_argument("--lsh_num_chunks_before", type=int, default=1)
    # model_args.add_argument("--lsh_num_chunks_after", type=int, default=0)
    # model_args.add_argument("--num_hashes", type=int, default=1)
    # model_args.add_argument("--num_buckets", default=None)
    # model_args.add_argument("--is_decoder", type=str2bool, default=False)
    # model_args.add_argument("--use_cache", type=str2bool, default=False)

    return parser


def get_args():
    parser = get_arg_parser()
    args = parser.parse_args()
    args.run_script = get_run_script()

    # random seed
    set_random_seed(args.random_seed)

    # settings
    if args.max_seq_size == 1024:
        args.pretrain_base_path += "_1023_128.pkl"
        args.axial_pos_shape = [32, 32]
        args.pretrain_train_batch_size = 64
        args.pretrain_test_batch_size = 128
        args.pretrain_max_num_evals = 20
        args.pretrain_update_steps = 5000
        args.finetune_train_batch_size = 64
        args.finetune_test_batch_size = 128
        args.finetune_max_num_evals = 500
        args.finetune_update_steps = 20
        args.finetune_patience = 30
    elif args.max_seq_size == 8192:
        args.pretrain_base_path += "_8191_1024.pkl"
        args.axial_pos_shape = [64, 128]
        args.pretrain_train_batch_size = 8
        args.pretrain_test_batch_size = 16
        args.pretrain_max_num_evals = 20
        args.pretrain_update_steps = 5000
        args.finetune_train_batch_size = 8
        args.finetune_test_batch_size = 16
        args.finetune_max_num_evals = 500
        args.finetune_update_steps = 20
        args.finetune_patience = 30

    # debug
    if args.debug_mode:
        args.num_workers = 0
        args.num_cross_folds = 1
        # args.interaction_base_path = f"/private/datasets/magneto_2021-01-27/user_interactions_wo_lecture_debug.pkl"
        args.interaction_base_path = (
            f"/private/datasets/magneto_2021-01-27/user_interactions_wo_lecture.pkl"
        )
        args.pretrain_base_path = args.pretrain_base_path.rstrip(".pkl") + "_debug.pkl"
        # args.score_base_path = (
        #     f"/private/datasets/magneto_2021-01-27/user_score_idxs_debug.pkl"
        # )
        args.score_base_path = (
            f"/private/datasets/magneto_2021-01-27/user_score_idxs.pkl"
        )
        args.pretrain_train_batch_size = 8
        args.pretrain_test_batch_size = 16
        args.pretrain_max_num_evals = 3
        args.pretrain_update_steps = 3
        args.finetune_train_batch_size = 8
        args.finetune_test_batch_size = 16
        args.finetune_max_num_evals = 3
        args.finetune_update_steps = 3
        args.wandb_name = "debug"

    # parse gpus
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        assert torch.cuda.is_available()
        args.device = "cuda"
    else:
        args.device = "cpu"

    # time output_func and loss sanity check
    if args.time_loss == "bce":
        assert args.time_output_func == "sigmoid"

    if (
        args.model == "electra"
        or args.model == "electra-reformer"
        or args.model == "electra-performer"
    ):
        args.targets = args.masked_features

    # wandb setting
    # input_masked_target
    args.wandb_name = f"{args.max_seq_size}_"
    # input
    if "choice" in args.input_features:
        args.wandb_name += "ch-"
    if "is_correct" in args.input_features:
        args.wandb_name += "ic-"
    if "elapsed_time" in args.input_features:
        args.wandb_name += "et-"
    if "is_on_time" in args.input_features:
        args.wandb_name += "it-"
    if "lag_time" in args.input_features:
        args.wandb_name += "lt-"
    if "exp_time" in args.input_features:
        args.wandb_name += "ext-"
    args.wandb_name = args.wandb_name.rstrip("-") + "_"
    # target
    if "choice" in args.input_features:
        args.wandb_name += "ch-"
    if "is_correct" in args.input_features:
        args.wandb_name += "ic-"
    if "elapsed_time" in args.input_features:
        args.wandb_name += "et-"
    if "is_on_time" in args.input_features:
        args.wandb_name += "it-"
    if "lag_time" in args.input_features:
        args.wandb_name += "lt-"
    if "exp_time" in args.input_features:
        args.wandb_name += "ext-"
    args.wandb_name = args.wandb_name.rstrip("-")

    # get weight path
    args.weight_path = f"{args.weight_base_path}/{args.model}/{args.wandb_name}"
    os.makedirs(args.weight_path, exist_ok=True)

    # wandb
    assert not (args.use_wandb and args.use_finetune_wandb)
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=args.wandb_tags,
            config=args,
        )
    elif args.use_finetune_wandb:
        assert args.train_mode == "finetune_only" or (
            args.train_mode == "finetune_only_from_pretrained_weight"
            and args.pretrained_weight_n_eval != -1
        )
        wandb.init(
            project=args.finetune_wandb_project,
            name=args.wandb_name,
            tags=args.wandb_tags,
            config=args,
        )

    return args, parser


ARGS, parser = get_args()
print_args(parser, ARGS)


class Const:
    UNKNOWN_PART = 8
    MAX_ELAPSED_TIME_IN_S = 300
    MAX_EXP_TIME_IN_S = 300
    MAX_LAG_TIME_IN_S = 86400

    PAD_VAL = 0
    FALSE_VAL = 1
    TRUE_VAL = 2

    CATE_VARS = ["qid", "part", "choice", "is_correct", "is_on_time"]
    CONT_VARS = ["elapsed_time", "exp_time", "lag_time", "lc", "rc"]

    FEATURE_SIZE = {
        "qid": None,  # initialized in get_q_info_dic
        "part": 8,  # 8 for unknown
        "choice": 4,  # a->1, b->2, c->3, d->4
        "is_correct": 2,
        "is_on_time": 2,
        "elapsed_time": None,  # continuous var
        "exp_time": None,  # continuous var
        "lag_time": None,  # continuous var
    }

    CLS_VAL = {
        "qid": None,  # initialized in get_q_info_dic
        "part": FEATURE_SIZE["part"] + 1,
        "choice": FEATURE_SIZE["choice"] + 1,
        "is_correct": FEATURE_SIZE["is_correct"] + 1,
        "is_on_time": FEATURE_SIZE["is_on_time"] + 1,
        "elapsed_time": 0,
        "exp_time": 0,
        "lag_time": 0,
    }

    MASK_VAL = {
        "qid": None,  # initialized in get_q_info_dic
        "part": FEATURE_SIZE["part"] + 2,
        "choice": FEATURE_SIZE["choice"] + 2,
        "is_correct": FEATURE_SIZE["is_correct"] + 2,
        "is_on_time": FEATURE_SIZE["is_on_time"] + 2,
        "elapsed_time": -1,
        "exp_time": -1,
        "lag_time": -1,
    }

    SCORE_SCALING_FACTOR = 495
