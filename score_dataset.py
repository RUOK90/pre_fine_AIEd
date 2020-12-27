import csv
import torch
import numpy as np
from torch.utils import data
from datetime import datetime

from config import *


def get_dataloaders(q_info_dic, score_base_path):
    user_inter_path = f"{score_base_path}/response"
    score_dataloaders = {}
    for cross_num in range(ARGS.num_cross_folds):
        print(f"processing score data split {cross_num}")
        train_user_path = f"{score_base_path}/{ARGS.num_cross_folds}fold/train_user_list_{cross_num}.csv"
        with open(train_user_path, "r") as f_r:
            train_user_id_list = np.array([line[0] for line in csv.reader(f_r)])
        val_user_path = f"{score_base_path}/{ARGS.num_cross_folds}fold/validation_user_list_{cross_num}.csv"
        with open(val_user_path, "r") as f_r:
            val_user_id_list = np.array([line[0] for line in csv.reader(f_r)])
        test_user_path = f"{score_base_path}/{ARGS.num_cross_folds}fold/test_user_list_{cross_num}.csv"
        with open(test_user_path, "r") as f_r:
            test_user_id_list = np.array([line[0] for line in csv.reader(f_r)])

        print(
            f"# train_users: {len(train_user_id_list)}, # val_users: {len(val_user_id_list)}, # test_users: {len(test_user_id_list)}"
        )

        train_dataset = ScoreDataSet(
            q_info_dic, user_inter_path, train_user_id_list, ARGS.aug_mode
        )
        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=ARGS.train_batch_size,
            shuffle=True,
            num_workers=ARGS.num_workers,
        )

        val_dataset = ScoreDataSet(
            q_info_dic, user_inter_path, val_user_id_list, "no_aug"
        )
        val_dataloader = data.DataLoader(
            dataset=val_dataset,
            batch_size=ARGS.test_batch_size,
            shuffle=False,
            num_workers=ARGS.num_workers,
        )

        test_dataset = ScoreDataSet(
            q_info_dic, user_inter_path, test_user_id_list, "no_aug"
        )
        test_dataloader = data.DataLoader(
            dataset=test_dataset,
            batch_size=ARGS.test_batch_size,
            shuffle=False,
            num_workers=ARGS.num_workers,
        )

        score_dataloaders[cross_num] = {
            "train": train_dataloader,
            "val": val_dataloader,
            "test": test_dataloader,
        }

    return score_dataloaders


class ScoreDataSet(data.Dataset):
    def __init__(self, q_info_dic, user_inter_path, user_id_list, aug_mode):
        self._q_info_dic = q_info_dic
        self._user_inter_path = user_inter_path
        self._user_id_list = user_id_list
        self._aug_mode = aug_mode
        self._user_inters, self._user_scores = get_user_data(
            q_info_dic, user_inter_path, user_id_list
        )

    def __len__(self):
        return len(self._user_id_list)

    def __getitem__(self, idx):
        uid = self._user_id_list[idx]
        processed_inters = preprocess_inters(
            self._user_inters[uid], self._user_scores[uid], self._aug_mode
        )
        return processed_inters


def get_user_data(q_info_dic, user_inter_path, user_id_list):
    uid2inters = {}
    uid2scores = {}

    for user_id in user_id_list:
        # get user score
        lc_score = int(user_id.split("_")[1])
        rc_score = int(user_id.split("_")[2])
        uid2scores[user_id] = {"lc": lc_score, "rc": rc_score}

        # get user inters
        with open(f"{user_inter_path}/{user_id}", "r") as f_r:
            inters = [line for line in csv.reader(f_r)]

        qid_list = []
        part_list = []
        is_correct_list = []
        elapsed_time_list = []
        is_on_time_list = []
        lag_time_list = []
        time_format = "%Y-%m-%d %H:%M:%S"
        last_time = None
        for inter in inters:
            timestamp, qid, choice, correct_ans, part, elapsed_time_in_ms = inter

            # qid
            qid = int(qid)
            if qid not in q_info_dic:
                continue
            else:
                qid = int(q_info_dic[qid]["dense_question_id"])

            # is_correct
            is_correct = Const.TRUE_VAL if choice == correct_ans else Const.FALSE_VAL

            # part
            part = Const.UNKNOWN_PART if part == "unknown" else int(part)

            # elasped_time, is_on_time
            elapsed_time_in_ms = float(elapsed_time_in_ms)
            time_limit_in_ms = float(
                q_info_dic[qid].get("time_limit_in_ms", Const.DEFAULT_TIME_LIMIT_IN_MS)
            )
            is_on_time = (
                Const.TRUE_VAL
                if elapsed_time_in_ms <= time_limit_in_ms
                else Const.FALSE_VAL
            )
            elapsed_time_in_s = elapsed_time_in_ms / 1000
            elapsed_time = min(elapsed_time_in_s / Const.MAX_ELAPSED_TIME_IN_S, 1)

            # lag_time
            curr_time = datetime.strptime(timestamp.split(".")[0], time_format)
            lag_time = (
                0
                if last_time is None
                else ((curr_time - last_time).total_seconds() - elapsed_time_in_s)
            )
            lag_time = min(lag_time / Const.MAX_LAG_TIME_IN_S, 1)
            last_time = curr_time

            qid_list.append(qid)
            part_list.append(part)
            is_correct_list.append(is_correct)
            is_on_time_list.append(is_on_time)
            elapsed_time_list.append(elapsed_time)
            lag_time_list.append(lag_time)

        uid2inters[user_id] = {
            "qid": np.array(qid_list),
            "part": np.array(part_list),
            "is_correct": np.array(is_correct_list),
            "is_on_time": np.array(is_on_time_list),
            "elapsed_time": np.array(elapsed_time_list),
            "lag_time": np.array(lag_time_list),
        }

    return uid2inters, uid2scores


def preprocess_inters(inters, scores, aug_mode):
    # get input features
    input_features = {name: np.array(inters[name]) for name in ARGS.input_features}

    # get labels
    labels = scores

    # get augmented input features and truncate to max seq size
    augmented_features = get_augmented_features(input_features, aug_mode)

    # get cls token appended augmented features
    cls_appended_features = {
        name: np.pad(feature, (1, 0), "constant", constant_values=Const.CLS_VAL[name])
        for name, feature in augmented_features.items()
    }

    # get zero padded input features
    padded_features, padding_masks = get_padded_features(
        cls_appended_features, return_padding_mask=True
    )

    return {
        "unmasked_feature": padded_features,
        "label": labels,
        "padding_mask": padding_masks,
    }


def get_augmented_features(features, aug_mode):
    if aug_mode == "no_aug":
        features = {
            name: feature[-ARGS.max_seq_size + 1 :]
            for name, feature in features.items()
        }
    elif aug_mode == "aug_only" or (
        aug_mode == "both" and np.random.random_sample() < ARGS.aug_sample_ratio
    ):
        seq_size = len(features["qid"])
        augmented_seq_size = int(seq_size * ARGS.aug_ratio)
        sampled_idxs = np.random.choice(seq_size, augmented_seq_size, replace=False)
        sorted_sampled_idxs = np.sort(sampled_idxs)

        features = {
            name: feature[sorted_sampled_idxs][-ARGS.max_seq_size + 1 :]
            for name, feature in features.items()
        }
    else:
        features = {
            name: feature[-ARGS.max_seq_size + 1 :]
            for name, feature in features.items()
        }

    return features


def get_padded_features(features, return_padding_mask):
    seq_size = len(features["qid"])
    num_pads = max(ARGS.max_seq_size - seq_size, 0)
    for name, feature in features.items():
        features[name] = np.pad(
            feature, (0, num_pads), "constant", constant_values=Const.PAD_VAL
        )

    padding_masks = None
    if return_padding_mask:
        padding_masks = np.pad(
            np.zeros(seq_size, dtype=bool),
            (0, num_pads),
            "constant",
            constant_values=True,
        )  # True: padding

    return features, padding_masks
