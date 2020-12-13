"""Main function for score estimation model fine-tuining
"""
import csv
import pandas as pd
import torch
import wandb
import numpy as np
import copy
import random
from torch.utils import data
import dill as pkl
from am_v2 import config, util, score_dataset, network, score_trainer, constant


def get_time_limit_in_ms(question_id, time_limits):
    """
    Load time limit data
    """
    return float(time_limits.get(str(question_id), constant.DEFAULT_TIME_LIMIT_IN_MS))


def get_user_data(base_path, old_to_new_item_id, user_id_list):
    """
    Load user / score / KT data
    """
    # Load time limit data from csv
    time_limits = {}
    with open("am_v2/tmp/time_limits.csv", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            question_id = row["question_id"]
            time_limit_in_ms = row["time_limit_in_ms"]
            time_limits[question_id] = time_limit_in_ms
    csv_file.close()

    content_data = {}
    with open("../final_tcr/payment_diagnosis_questions.csv", newline="") as content_file:
        reader = csv.DictReader(content_file)
        for row in reader:
            content_data[row['question_id']] = {
                "correct_answer": row["correct_answer"],
                "part": row["part"],
            }

    content_file.close()

    with open("../final_tcr/user_score_dict.pkl", 'rb') as score_file:
        user_score_dict = pkl.load(score_file)
    
    uid2sequences = {}
    uid2scores = {}
    for user_id in user_id_list:
        user_file_path = f'{base_path}/{user_id}'
        total_score = user_score_dict[user_id]
        lc_score = total_score//2
        rc_score = total_score - lc_score
        
        uid2scores[user_id] = {'lc': lc_score, 'rc': rc_score}
        with open(user_file_path, "r") as file:
            line_list = file.readlines()

        qid_list = []
        part_list = []
        is_correct_list = []
        is_on_time_list = []
        lag_time_list = []
        elapsed_time_list = []
        max_elapsed_time = 300
        max_lag_time = 86400
        last_time = None
        for line in line_list[1:-1]:
            line = line.split(",")
            old_qid = line[1]
            if int(old_qid[1:]) not in old_to_new_item_id:
                continue
            qid = old_to_new_item_id[int(old_qid[1:])]
            user_answer = line[2]
            correct_answer = content_data[old_qid]["correct_answer"]
            part = content_data[old_qid]["part"]
            if part == "unknown":
                part = 8
            else:
                part = int(part)
            if user_answer == correct_answer:
                is_correct = constant.TRUE_INDEX
            else:
                is_correct = constant.FALSE_INDEX

            elapsed_time_in_ms = float(line[3])
            time_limit_in_ms = get_time_limit_in_ms(old_qid, time_limits)
            is_on_time = (
                constant.TRUE_INDEX
                if elapsed_time_in_ms <= time_limit_in_ms
                else constant.FALSE_INDEX
            )
            elapsed_time = elapsed_time_in_ms / 1000
            curr_time = int(line[0])
            if last_time is None:
                lag_time = 0
            else:
                lag_time = (curr_time - last_time) / 1000 - elapsed_time
            last_time = curr_time

            elapsed_time = min(elapsed_time / max_elapsed_time, 1)
            lag_time = max(0, min(lag_time / max_lag_time, 1))
            # lag_time = min(lag_time / max_lag_time, 1)
            qid_list.append(qid)
            part_list.append(part)
            is_correct_list.append(is_correct)
            is_on_time_list.append(is_on_time)
            elapsed_time_list.append(elapsed_time)
            lag_time_list.append(lag_time)
        
        file.close()
        
        uid2sequences[user_id]={
            'qid': qid_list,
            'part': part_list,
            'is_correct': is_correct_list,
            'is_on_time': is_on_time_list,
            'elapsed_time': elapsed_time_list,
            'lag_time': lag_time_list
        }
        # print(uid2sequences[user_id])
    return uid2sequences, uid2scores


class DataSet(data.Dataset):
    """
    DataSet class for score prediction data

    Args:
        is_training: training mode
        is_augmented: to random sample or not (do in training)
        sample_rate: how much to random sample
        base_path: base path for data loading
        content_mapping: sequential mapping for questions
        user_id_list: list of user ids
        sequence_size: maximum size of sequence
    """

    def __init__(
        self,
        is_training,
        is_augmented,
        sample_rate,
        base_path,
        content_mapping,
        user_id_list,
        sequence_size,
    ):
        self._is_training = is_training
        self._is_augmented = is_augmented
        self._sample_rate = sample_rate
        self._user_id_list = user_id_list
        self._sequences, self._scores = get_user_data(
            base_path, content_mapping, user_id_list
        )

        self._sequence_size = sequence_size

    @property
    def the_number_of_users(self):
        """
        Size of dataset in terms of users
        """
        return len(self._user_id_list)

    def _get_augmented_x(
        self, input_features,
    ):
        """
        Gets augmented input for batching
        Creates random sampled partitions
        """

        index_list = [e for e in range(len(input_features['qid']))]
        num_sample = int(len(index_list) * self._sample_rate)
        index_list = random.sample(index_list, num_sample)
        index_list.sort()
        index_list = np.array(index_list, dtype=int)

        aug_features = {
            name: np.array(feature)[index_list].tolist()
            for name, feature in input_features.items()
        }
        return aug_features

    def _get_x(self, user_id):
        """
        Gets single input for batching
        During training, use augmented input (random samplin)
        During testing, use last few samples
        """
        input_features = self._sequences[user_id]

        if self._is_augmented:
            features = self._get_augmented_x(input_features)
        else:
            features = input_features

        num_pads = max(0, self._sequence_size - len(features['qid']))
        zero_pads = [0] * num_pads
        features = {
            name: torch.Tensor(
                feature[-self._sequence_size:] + zero_pads
            )
            for name, feature in features.items()
        }
        return features

    def __repr__(self):
        return (
            f"Training mode: {self._is_training}, # of users:"
            f" {self.the_number_of_users}"
        )

    def __len__(self):
        """Length of dataset"""
        return len(self._user_id_list)

    def __getitem__(self, index):
        user_id = self._user_id_list[index]
        # count = self._sequences[user_id]["count"]
        features = self._get_x(user_id)
        lc_score = self._scores[user_id]["lc"]
        rc_score = self._scores[user_id]["rc"]
        return (
            features,
            lc_score,
            rc_score,
        )


def get_score_generators(content_mapping):
    """
    Main code for fine-tuning
    Can be called during pre-training for validation
    """
    # set
    args = config.ARGS

    base_path = "../final_tcr"
    user_path = f"{base_path}/users"

    train_user_id_list = pd.read_csv(f'{base_path}/user_split/train_users_list.csv', header=None).values.flatten()
    val_user_id_list = pd.read_csv(f'{base_path}/user_split/val_users_list.csv', header=None).values.flatten()
    test_user_id_list = pd.read_csv(f'{base_path}/user_split/test_users_list.csv', header=None).values.flatten()
    print(
        f"# of train_users: {len(train_user_id_list)}, # of val_users:"
        f" {len(val_user_id_list)}, # of test_users: {len(test_user_id_list)}"
    )

    train_data = DataSet(
        True,
        args.is_augmented,
        args.sample_rate,
        user_path,
        content_mapping,
        train_user_id_list,
        args.seq_size,
    )
    val_data = DataSet(
        False, False, 1, user_path, content_mapping, val_user_id_list, args.seq_size,
    )
    test_data = DataSet(
        False, False, 1, user_path, content_mapping, test_user_id_list, args.seq_size,
    )

    print(train_data)
    print(val_data)
    print(test_data)

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
    score_generators = (train_generator, val_generator, test_generator)

    return score_generators


def finetune_model(
        _q_size,
        _score_generators,
        _pretrain_weight_path,
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

    util.load_pretrained_weight(
        _pretrain_weight_path,
        model,
        config.ARGS.device
    )

    trainer = score_trainer.Trainer(
        copy.deepcopy(model),
        config.ARGS.debug_mode,
        config.ARGS.num_epochs if _num_epochs is None else _num_epochs,
        config.ARGS.weight_path,
        config.ARGS.d_model,
        config.ARGS.warmup_steps,
        config.ARGS.lr,
        config.ARGS.device,
        _score_generators,
        config.ARGS.lambda1,
        config.ARGS.lambda2,
    )
    trainer.train()
    val_min_mae = trainer._validation_min_mae
    test_mae = trainer._test_mae
    print(
        f"└──────[best] val MAE: {trainer._validation_min_mae}, "
        f"dev epoch: {trainer._validation_min_mae_epoch}"
    )

    return test_mae, val_min_mae,


def main(content_mapping=None):
    args = config.ARGS
    # fix random seeds
    content_mapping = util.read_mapping_item_id(f"am_v2/load/ednet_content_mapping.csv")
    q_size = len(content_mapping)
    if not args.debug_mode:
        wandb.init(project=args.project, name=args.name, tags=args.tags, config=args)

    score_generators = get_score_generators(content_mapping)
    for epoch in range(100):
        test_mae, val_min_mae = finetune_model(
            q_size,
            score_generators,
            f'weight/201107-ET+LT-T+C-3.7.6-1.6.0/{epoch}.pt',
        )

        if not args.debug_mode:
            wandb.log(
                {
                    "Val MAE": val_min_mae,
                    "Test MAE": test_mae,
                }
            )


if __name__ == "__main__":
    main()
