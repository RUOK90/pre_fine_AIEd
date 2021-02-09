import pickle
import numpy as np
import torch
from torch.utils import data

from config import *


def get_dataloaders(q_info_dic, user_inters_dic, pretrain_base_path):
    print(f"processing pretrain data")
    with open(pretrain_base_path, "rb") as f_r:
        user_inter_windows = pickle.load(f_r)
    print(
        f"# train: {len(user_inter_windows['train'])}, # val: {len(user_inter_windows['val'])}"
    )

    train_dataset = PretrainDataSet(
        q_info_dic, user_inters_dic, user_inter_windows["train"]
    )
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=ARGS.pretrain_train_batch_size,
        shuffle=True,
        num_workers=ARGS.num_workers,
    )

    val_dataset = PretrainDataSet(
        q_info_dic, user_inters_dic, user_inter_windows["val"]
    )
    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=ARGS.pretrain_test_batch_size,
        shuffle=False,
        num_workers=ARGS.num_workers,
    )

    pretrain_dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
    }

    return pretrain_dataloaders


class PretrainDataSet(data.Dataset):
    def __init__(self, q_info_dic, user_inters_dic, user_inter_windows):
        self._q_info_dic = q_info_dic
        self._user_inters_dic = user_inters_dic
        self._user_inter_windows = user_inter_windows

    def __len__(self):
        return len(self._user_inter_windows)

    def __getitem__(self, idx):
        uid, start_idx, end_idx = self._user_inter_windows[idx]
        processed_inters = preprocess_inters(
            self._q_info_dic, self._user_inters_dic[uid][start_idx:end_idx]
        )
        return processed_inters


def preprocess_inters(q_info_dic, inters):
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

    # get cls token appended all_features
    cls_appended_all_features = {
        name: np.pad(feature, (1, 0), "constant", constant_values=Const.CLS_VAL[name])
        for name, feature in all_features.items()
    }

    # get unmasked input features
    unmasked_features = {
        name: cls_appended_all_features[name] for name in ARGS.input_features
    }

    # get masked input features
    masked_features, input_masks = get_masked_input_features(unmasked_features)

    # get labels
    labels = get_labels(cls_appended_all_features)

    # get zero padded unmasked input features
    padded_unmasked_features, padding_masks = get_padded_features(
        unmasked_features, return_padding_mask=True
    )

    # get zero padded masked input features
    padded_masked_features, _ = get_padded_features(
        masked_features, return_padding_mask=False
    )

    # get zero padded input masks
    padded_input_masks = get_padded_masks(input_masks)

    # get zero padded labels
    padded_labels, _ = get_padded_features(labels, return_padding_mask=False)

    seq_size = len(list(cls_appended_all_features.values())[0])

    return {
        "unmasked_feature": padded_unmasked_features,
        "masked_feature": padded_masked_features,
        "label": padded_labels,
        "input_mask": padded_input_masks,
        "padding_mask": padding_masks,
        "seq_size": seq_size,
    }


def get_masked_input_features(features):
    masked_input_features = {}
    seq_size = len(list(features.values())[0])
    masks = np.random.random_sample(seq_size) < ARGS.random_mask_ratio  # True: mask
    for name, feature in features.items():
        if name in ARGS.masked_features:
            masked_feature = feature * (1 - masks) + masks * Const.MASK_VAL[name]
            masked_input_features[name] = masked_feature
        else:
            masked_input_features[name] = feature

    return masked_input_features, masks


def get_labels(features):
    labels = {}
    for name, feature in features.items():
        if name in ARGS.targets:
            if name in Const.CATE_VARS:
                labels[name] = feature - 1
            elif name in Const.CONT_VARS:
                labels[name] = feature

    return labels


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


def get_padded_masks(masks):
    seq_size = len(masks)
    num_pads = max(ARGS.max_seq_size - seq_size, 0)
    padded_masks = np.pad(
        masks, (0, num_pads), "constant", constant_values=Const.PAD_VAL
    )

    return padded_masks
