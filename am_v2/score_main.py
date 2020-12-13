"""Main function for score estimation model fine-tuining
"""
import csv
import pandas as pd
import torch
import wandb
import numpy as np
import copy
import dill as pkl
from am_v2 import config, util, score_dataset, network, score_trainer


def finetune_model(
    _q_size,
    _pretrain_weight_path,
    _score_generators,
    _num_epochs=None,
):
    torch.manual_seed(0)
    np.random.seed(0)
    """Run pre-trained model"""
    model = network.ScoreModel(
        q_size=_q_size,
        t_size=3,
        r_size=3,
        p_size=7,
        n_layer=config.ARGS.num_layer,
        d_model=config.ARGS.d_model,
        h=config.ARGS.num_heads,
        dropout=config.ARGS.dropout,
        device=config.ARGS.device,
        seq_size=config.ARGS.seq_size,
        is_feature_based=config.ARGS.is_feature_based,
    ).to(config.ARGS.device)
    # print(model)
    val_min_maes = {}
    test_maes = {}
    for cross_num, score_gen in _score_generators.items():

        if config.ARGS.is_pretrain:
            if config.ARGS.load_question_embed in ['bert', 'quesnet', 'word2vec']:
                print(f"Loading {_pretrain_weight_path}")
                with open(_pretrain_weight_path, 'rb') as saved:
                    qid_to_vec = pkl.load(saved)
                    if config.ARGS.load_question_embed == 'bert':
                        embedding = torch.stack([qid_to_vec[qid].detach() for qid in range(1, len(qid_to_vec))]).to(config.ARGS.device)
                    else:
                        embedding = torch.stack([torch.from_numpy(qid_to_vec[qid]).to(config.ARGS.device) for qid in range(1, len(qid_to_vec))])
                    model.embedding_by_feature['qid'].weight = torch.nn.Parameter(embedding)
            else:
                # pretrain_weight_path = f"{config.ARGS.weight_path}/{_pretrain_epoch}.pt"
                util.load_pretrained_weight(_pretrain_weight_path, model, config.ARGS.device)

        if config.ARGS.freeze_embedding is True:
            model.freeze_embed()
        if config.ARGS.freeze_encoder_block is True:
            model.freeze_encoder()

        trainer = score_trainer.Trainer(
            copy.deepcopy(model),
            config.ARGS.debug_mode,
            config.ARGS.num_epochs if _num_epochs is None else _num_epochs,
            config.ARGS.weight_path,
            config.ARGS.d_model,
            config.ARGS.warmup_steps,
            config.ARGS.lr,
            config.ARGS.device,
            score_gen,
            config.ARGS.lambda1,
            config.ARGS.lambda2,
        )
        trainer.train()
        val_min_maes[cross_num] = trainer._validation_min_mae
        test_maes[cross_num] = trainer._test_mae
        print(
            f"└──────[best] dev MAE: {trainer._validation_min_mae}, "
            f"dev epoch: {trainer._validation_min_mae_epoch}"
        )

    return test_maes, val_min_maes,


def get_score_generators(content_mapping=None):
    """
    Main code for fine-tuning
    Can be called during pre-training for validation
    """
    # set
    args = config.ARGS

    base_path = "/dataset/score_data/14d_10q"
    user_path = f"{base_path}/response"
    if args.use_new_score_data:
        base_path = "/shared/new_score_data/"
        user_path = base_path

    if content_mapping is None:
        mapping_file_path = f"am_v2/load/new_content_dict.csv"
        # mapping_file_path = f"am_v2/load/deployed_content_dict.csv"
        content_mapping = util.read_mapping_item_id(mapping_file_path)

    score_generators = {}
    num_folds = args.num_cross_folds
    for cross_num in range(num_folds):
        print(f"=====[ score_data_split {cross_num} ]=====")
        train_user_id_list = pd.read_csv(f'{base_path}/{num_folds}fold/train_user_list_{cross_num}.csv', header=None).values.flatten()
        val_user_id_list = pd.read_csv(f'{base_path}/{num_folds}fold/validation_user_list_{cross_num}.csv', header=None).values.flatten()
        test_user_id_list = pd.read_csv(f'{base_path}/{num_folds}fold/test_user_list_{cross_num}.csv', header=None).values.flatten()
        print(
            f"# of train_users: {len(train_user_id_list)}, # of val_users:"
            f" {len(val_user_id_list)}, # of test_users: {len(test_user_id_list)}"
        )

        train_data = score_dataset.DataSet(
            True,
            args.is_augmented,
            args.sample_rate,
            user_path,
            content_mapping,
            train_user_id_list,
            args.seq_size,
        )
        val_data = score_dataset.DataSet(
            False, False, 1, user_path, content_mapping, val_user_id_list, args.seq_size,
        )
        test_data = score_dataset.DataSet(
            False, False, 1, user_path, content_mapping, test_user_id_list, args.seq_size,
        )

        # print(train_data)
        # print(val_data)
        # print(test_data)

        train_generator = torch.utils.data.DataLoader(
            dataset=train_data,
            shuffle=True,
            batch_size=args.train_batch,
            num_workers=args.num_workers,
        )
        val_generator = torch.utils.data.DataLoader(
            dataset=val_data,
            shuffle=False,
            batch_size=args.test_batch,
            num_workers=args.num_workers,
        )
        test_generator = torch.utils.data.DataLoader(
            dataset=test_data,
            shuffle=False,
            batch_size=args.test_batch,
            num_workers=args.num_workers,
        )
        score_generators[cross_num] = (train_generator, val_generator, test_generator)

    return score_generators


def main(content_mapping=None):
    args = config.ARGS
    # fix random seeds
    content_mapping = util.read_mapping_item_id(f"am_v2/load/ednet_content_mapping.csv")
    q_size = len(content_mapping)
    if config.ARGS.load_question_embed in ['bert', 'quesnet', 'word2vec']:
        q_size = 19398
    if not args.debug_mode:
        wandb.init(project=args.project, name=args.name, tags=args.tags, config=args)

    score_generators = get_score_generators(content_mapping)

    if args.pretrain_number == -1:
        pretrain_epochs = range(args.pretrain_epoch_start_index, args.pretrain_epoch_end_index)

        for pretrain_epoch in pretrain_epochs:
            if args.load_question_embed == 'bert':
                path = f"/dev/shm/BERT/qid_to_vector_{pretrain_epoch}.pkl"
            elif args.load_question_embed == 'quesnet':
                path = f"/dev/shm/quesnet/quesnet_weight_{pretrain_epoch}_vectors.pkl"
            elif args.load_question_embed == 'word2vec':
                path = f"/dev/shm/word2vec/word2vec_{pretrain_epoch}.pkl"
            else:
                path = f"{args.weight_path}/{pretrain_epoch}.pt"
            print("└───[Val] Score data dev set fine-tuning evaluation")
            test_maes, val_min_maes = finetune_model(
                q_size,
                path,
                score_generators,
            )
            average_val_min_mae = sum(val_min_maes.values()) / len(val_min_maes)
            average_test_mae = sum(test_maes.values()) / len(test_maes)
            print(f'5-means dev mae:{sum(val_min_maes.values()) / len(val_min_maes)}')

            if not args.debug_mode:
                wandb.log(
                    {
                        "5-mean Dev MAE": average_val_min_mae,
                        "5-mean Test MAE": average_test_mae,
                        "Dev MAE 0": val_min_maes[0],
                        "Dev MAE 1": val_min_maes[1],
                        "Dev MAE 2": val_min_maes[2],
                        "Dev MAE 3": val_min_maes[3],
                        "Dev MAE 4": val_min_maes[4],
                        # "Dev MAE 5": val_min_maes[5],
                        # "Dev MAE 6": val_min_maes[6],
                        # "Dev MAE 7": val_min_maes[7],
                        # "Dev MAE 8": val_min_maes[8],
                        # "Dev MAE 9": val_min_maes[9],
                        "Test MAE 0": test_maes[0],
                        "Test MAE 1": test_maes[1],
                        "Test MAE 2": test_maes[2],
                        "Test MAE 3": test_maes[3],
                        "Test MAE 4": test_maes[4],
                        # "Test MAE 5": test_maes[5],
                        # "Test MAE 6": test_maes[6],
                        # "Test MAE 7": test_maes[7],
                        # "Test MAE 8": test_maes[8],
                        # "Test MAE 9": test_maes[9],
                }
            )
    elif args.pretrain_number == -2:
        test_maes, val_min_maes = finetune_model(
            q_size,
            None,
            score_generators,
        )
        average_val_min_mae = sum(val_min_maes.values()) / len(val_min_maes)
        average_test_mae = sum(test_maes.values()) / len(test_maes)
        print(f'5-means dev mae:{sum(val_min_maes.values()) / len(val_min_maes)}')

        if not args.debug_mode:
            wandb.log(
                {
                    "5-mean Dev MAE": average_val_min_mae,
                    "5-mean Test MAE": average_test_mae,
                    "Dev MAE 0": val_min_maes[0],
                    "Dev MAE 1": val_min_maes[1],
                    "Dev MAE 2": val_min_maes[2],
                    "Dev MAE 3": val_min_maes[3],
                    "Dev MAE 4": val_min_maes[4],
                    "Test MAE 0": test_maes[0],
                    "Test MAE 1": test_maes[1],
                    "Test MAE 2": test_maes[2],
                    "Test MAE 3": test_maes[3],
                    "Test MAE 4": test_maes[4]
                }
            )

    else:
        pretrain_weight_path = f"{args.weight_path}/{args.pretrain_number}.pt"
        test_maes, val_min_maes = finetune_model(
            len(content_mapping),
            pretrain_weight_path,
            score_generators,
        )
        average_val_min_mae = sum(val_min_maes.values()) / len(val_min_maes)
        average_test_mae = sum(test_maes.values()) / len(test_maes)
        print(f'10-means dev mae:{sum(val_min_maes.values()) / len(val_min_maes)}')

        if not args.debug_mode:
            wandb.log(
                {
                    "10-mean Dev MAE": average_val_min_mae,
                    "10-mean Test MAE": average_test_mae,
                    "Dev MAE 0": val_min_maes[0],
                    "Dev MAE 1": val_min_maes[1],
                    "Dev MAE 2": val_min_maes[2],
                    "Dev MAE 3": val_min_maes[3],
                    "Dev MAE 4": val_min_maes[4],
                    "Dev MAE 5": val_min_maes[5],
                    "Dev MAE 6": val_min_maes[6],
                    "Dev MAE 7": val_min_maes[7],
                    "Dev MAE 8": val_min_maes[8],
                    "Dev MAE 9": val_min_maes[9],
                    "Test MAE 0": test_maes[0],
                    "Test MAE 1": test_maes[1],
                    "Test MAE 2": test_maes[2],
                    "Test MAE 3": test_maes[3],
                    "Test MAE 4": test_maes[4],
                    "Test MAE 5": test_maes[5],
                    "Test MAE 6": test_maes[6],
                    "Test MAE 7": test_maes[7],
                    "Test MAE 8": test_maes[8],
                    "Test MAE 9": test_maes[9],
                }
            )


if __name__ == "__main__":
    main()
