import pickle
import numpy as np
import torch
from torch.utils import data

from config import *


def get_dataloaders(q_info_dic, user_inters_dic, score_base_path):
    print(f"processing score data")
    with open(score_base_path, "rb") as f_r:
        user_score_idxs = pickle.load(f_r)

    score_dataloaders = {}
    for cross_num in range(ARGS.num_cross_folds):
        print(
            f"{cross_num} fold, # train: {len(user_score_idxs[cross_num]['train'])}, # val: {len(user_score_idxs[cross_num]['val'])}, # test: {len(user_score_idxs[cross_num]['test'])}"
        )

        n_train = int(
            len(user_score_idxs[cross_num]["train"]) * ARGS.finetune_train_ratio
        )
        train_dataset = ScoreDataSet(
            q_info_dic,
            user_inters_dic,
            user_score_idxs[cross_num]["train"][:n_train],
            ARGS.aug_mode,
        )
        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=ARGS.finetune_train_batch_size,
            shuffle=True,
            num_workers=ARGS.num_workers,
        )

        val_dataset = ScoreDataSet(
            q_info_dic, user_inters_dic, user_score_idxs[cross_num]["val"], "no_aug"
        )
        val_dataloader = data.DataLoader(
            dataset=val_dataset,
            batch_size=ARGS.finetune_test_batch_size,
            shuffle=False,
            num_workers=ARGS.num_workers,
        )

        test_dataset = ScoreDataSet(
            q_info_dic, user_inters_dic, user_score_idxs[cross_num]["test"], "no_aug"
        )
        test_dataloader = data.DataLoader(
            dataset=test_dataset,
            batch_size=ARGS.finetune_test_batch_size,
            shuffle=False,
            num_workers=ARGS.num_workers,
        )

        all_dataset = ScoreDataSet(
            q_info_dic,
            user_inters_dic,
            user_score_idxs[cross_num]["train"][:n_train]
            + user_score_idxs[cross_num]["val"]
            + user_score_idxs[cross_num]["test"],
            "no_aug",
        )
        all_dataloader = data.DataLoader(
            dataset=all_dataset,
            batch_size=ARGS.finetune_test_batch_size,
            shuffle=False,
            num_workers=ARGS.num_workers,
        )

        score_dataloaders[cross_num] = {
            "train": train_dataloader,
            "val": val_dataloader,
            "test": test_dataloader,
            "all": all_dataloader,
        }

    return score_dataloaders


class ScoreDataSet(data.Dataset):
    def __init__(self, q_info_dic, user_inters_dic, user_score_idxs, aug_mode):
        self._q_info_dic = q_info_dic
        self._user_inters_dic = user_inters_dic
        self._user_score_idxs = user_score_idxs
        self._aug_mode = aug_mode

    def __len__(self):
        return len(self._user_score_idxs)

    def __getitem__(self, idx):
        uid, end_idx, lc, rc = self._user_score_idxs[idx]
        processed_inters = preprocess_inters(
            self._q_info_dic,
            self._user_inters_dic[uid][:end_idx],
            lc,
            rc,
            self._aug_mode,
        )
        return processed_inters


def preprocess_inters(q_info_dic, inters, lc, rc, aug_mode):
    all_features = {
        "qid": np.array([q_info_dic[q] for q in inters["item_id"]]),
        "part": inters["part_id"],
        "choice": inters["user_answer"],
        "is_correct": np.array([ic + 1 for ic in inters["is_correct"]]),
        "is_on_time": np.array([it + 1 for it in inters["timeliness"]]),
        "elapsed_time": np.array(
            [
                max(0, min(et / Const.MAX_ELAPSED_TIME_IN_S, 1))
                for et in inters["elapsed_time_in_s"]
            ]
        ).astype(np.float64),
        "exp_time": np.array(
            [
                max(0, min(et / Const.MAX_EXP_TIME_IN_S, 1))
                for et in inters["exp_time_in_s"]
            ]
        ).astype(np.float64),
        "lag_time": np.array(
            [
                max(0, min(lt / Const.MAX_LAG_TIME_IN_S, 1))
                for lt in inters["lag_time_in_s"]
            ]
        ).astype(np.float64),
    }

    # get input features
    input_features = {name: all_features[name] for name in ARGS.input_features}

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

    # get labels
    labels = {"lc": lc, "rc": rc}

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
        seq_size = len(list(features.values())[0])
        if ARGS.aug_ratio == -1:
            aug_ratio = np.random.random_sample() * 0.8 + 0.1
        else:
            aug_ratio = ARGS.aug_ratio
        augmented_seq_size = int(seq_size * aug_ratio)
        sampled_idxs = np.random.choice(seq_size, augmented_seq_size, replace=False)
        sorted_sampled_idxs = np.sort(sampled_idxs)

        features = {
            name: feature[sorted_sampled_idxs][-ARGS.max_seq_size + 1 :]
            for name, feature in features.items()
        }
    else:  # both but not aug
        features = {
            name: feature[-ARGS.max_seq_size + 1 :]
            for name, feature in features.items()
        }

    return features


def get_padded_features(features, return_padding_mask):
    seq_size = len(list(features.values())[0])
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
