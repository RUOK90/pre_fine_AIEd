import torch
import wandb
import pandas as pd
import numpy as np
import dill as pkl
import copy
from am_v2 import util
from am_v2 import config
from am_v2 import review_dataset, review_trainer
from am_v2 import network


def finetune_model(
    _q_size,
    _pretrain_weight_path,
    _review_generators,
    _num_epochs=None,
):
    torch.manual_seed(0)
    np.random.seed(0)
    """Run pre-trained model"""
    model = network.ReviewModel(
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

    val_max_aucs = {}
    test_aucs = {}
    for cross_num, review_gen in _review_generators.items():

        if config.ARGS.is_pretrain:
            if config.ARGS.load_question_embed in ['bert', 'quesnet', 'word2vec']:
                print(f"Loading {_pretrain_weight_path}")
                with open(_pretrain_weight_path, 'rb') as saved:
                    qid_to_vec = pkl.load(saved)
                    if config.ARGS.load_question_embed == 'bert':
                        embedding = torch.stack([qid_to_vec[qid].detach() for qid in range(1, len(qid_to_vec))])
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

        trainer = review_trainer.Trainer(
            copy.deepcopy(model),
            config.ARGS.debug_mode,
            config.ARGS.num_epochs if _num_epochs is None else _num_epochs,
            config.ARGS.weight_path,
            config.ARGS.d_model,
            config.ARGS.warmup_steps,
            config.ARGS.lr,
            config.ARGS.device,
            review_gen,
            config.ARGS.lambda1,
            config.ARGS.lambda2,
        )
        trainer.train()
        val_max_aucs[cross_num] = trainer._validation_max_auc
        test_aucs[cross_num] = trainer._test_auc
        print(
            f"└──────[best] dev review AUC: {trainer._validation_max_auc}, "
            f"dev epoch: {trainer._validation_max_auc_epoch}"
        )

    return test_aucs, val_max_aucs,


def get_review_generators(content_mapping):
    """
    Main code for fine-tuning
    Can be called during pre-training for validation
    """
    # set
    args = config.ARGS

    base_path = "/dev/shm/review_data"
    user_path = f"{base_path}/response"

    review_generators = {}
    num_folds = args.num_cross_folds
    for cross_num in range(num_folds):
        print(f"=====[ review_data_split {cross_num} ]=====")
        train_user_id_list = pd.read_csv(f'{base_path}/{num_folds}fold/train_user_list_{cross_num}.csv', header=None).values.flatten()
        val_user_id_list = pd.read_csv(f'{base_path}/{num_folds}fold/validation_user_list_{cross_num}.csv', header=None).values.flatten()
        test_user_id_list = pd.read_csv(f'{base_path}/{num_folds}fold/test_user_list_{cross_num}.csv', header=None).values.flatten()
        print(
            f"# of train_users: {len(train_user_id_list)}, # of val_users:"
            f" {len(val_user_id_list)}, # of test_users: {len(test_user_id_list)}"
        )

        train_data = review_dataset.DataSet(
            True,
            user_path,
            content_mapping,
            train_user_id_list,
            args.seq_size,
        )
        val_data = review_dataset.DataSet(
            False, user_path, content_mapping, val_user_id_list, args.seq_size,
        )
        test_data = review_dataset.DataSet(
            False, user_path, content_mapping, test_user_id_list, args.seq_size,
        )

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
        review_generators[cross_num] = (train_generator, val_generator, test_generator)

    return review_generators


def main(content_mapping=None):
    args = config.ARGS
    # fix random seeds
    content_mapping = util.read_mapping_item_id(f"am_v2/load/ednet_content_mapping.csv")
    q_size = len(content_mapping)
    if config.ARGS.load_question_embed in ['bert', 'quesnet', 'word2vec']:
        q_size = 19398
    if not args.debug_mode:
        wandb.init(project=args.project, name=args.name, tags=args.tags, config=args)

    review_generators = get_review_generators(content_mapping)

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
            print("└───[Val] Review data dev set fine-tuning evaluation")
            test_aucs, val_max_aucs = finetune_model(
                q_size,
                path,
                review_generators,
            )
            average_val_max_auc = sum(val_max_aucs.values()) / len(val_max_aucs)
            average_test_auc = sum(test_aucs.values()) / len(test_aucs)
            print(f'5-means dev review auc:{sum(val_max_aucs.values()) / len(val_max_aucs)}')

            if not args.debug_mode:
                wandb.log(
                    {
                        "5-mean Dev review AUC": average_val_max_auc,
                        "5-mean Test review AUC": average_test_auc,
                        "Dev review AUC 0": val_max_aucs[0],
                        "Dev review AUC 1": val_max_aucs[1],
                        "Dev review AUC 2": val_max_aucs[2],
                        "Dev review AUC 3": val_max_aucs[3],
                        "Dev review AUC 4": val_max_aucs[4],
                        "Test review AUC 0": test_aucs[0],
                        "Test review AUC 1": test_aucs[1],
                        "Test review AUC 2": test_aucs[2],
                        "Test review AUC 3": test_aucs[3],
                        "Test review AUC 4": test_aucs[4],
                }
            )

    elif args.pretrain_number == -2:
        test_aucs, val_max_aucs = finetune_model(
            q_size,
            None,
            review_generators,
        )
        average_val_max_auc = sum(val_max_aucs.values()) / len(val_max_aucs)
        average_test_auc = sum(test_aucs.values()) / len(test_aucs)
        print(f'5-means dev review auc:{sum(val_max_aucs.values()) / len(val_max_aucs)}')

        if not args.debug_mode:
            wandb.log(
                {
                    "5-mean Dev review AUC": average_val_max_auc,
                    "5-mean Test review AUC": average_test_auc,
                    "Dev review AUC 0": val_max_aucs[0],
                    "Dev review AUC 1": val_max_aucs[1],
                    "Dev review AUC 2": val_max_aucs[2],
                    "Dev review AUC 3": val_max_aucs[3],
                    "Dev review AUC 4": val_max_aucs[4],
                    "Test review AUC 0": test_aucs[0],
                    "Test review AUC 1": test_aucs[1],
                    "Test review AUC 2": test_aucs[2],
                    "Test review AUC 3": test_aucs[3],
                    "Test review AUC 4": test_aucs[4]
                }
            )

if __name__ == "__main__":
    main()
