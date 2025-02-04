import pickle as pkl
import numpy as np
import torch
from torch.utils import data

from config import *


def get_dataloaders(q_info_dic, pretrain_base_path):
    print(f"processing pretrain data")
    user_window_path = f"{pretrain_base_path}/ednet_sample_list_50.pkl"
    with open(user_window_path, "rb") as f_r:
        train_user_windows, val_user_windows, test_user_windows = pkl.load(f_r)
        train_user_windows = np.array(train_user_windows)
        val_user_windows = np.array(val_user_windows)

    train_user_inter_path = f"{pretrain_base_path}/ednet_train_sequences.pkl"
    train_dataset = EdNetDataSet(q_info_dic, train_user_inter_path, train_user_windows)
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=ARGS.pretrain_train_batch_size,
        shuffle=True,
        num_workers=ARGS.num_workers,
    )

    val_user_inter_path = f"{pretrain_base_path}/ednet_dev_sequences.pkl"
    val_dataset = EdNetDataSet(q_info_dic, val_user_inter_path, val_user_windows)
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


class EdNetDataSet(data.Dataset):
    def __init__(self, q_info_dic, user_inter_path, user_windows):
        self._q_info_dic = q_info_dic
        self._user_inter_path = user_inter_path
        self._user_windows = user_windows
        self._user_inters = get_user_data(user_inter_path)

    def __len__(self):
        return len(self._user_windows)

    def __getitem__(self, idx):
        uid, window_idx = self._user_windows[idx]
        processed_inters = preprocess_inters(self._user_inters[uid], window_idx)
        return processed_inters


def get_user_data(user_inter_path):
    with open(user_inter_path, "rb") as f_r:
        uid2inters = pkl.load(f_r)

    for uid, inters in uid2inters.items():
        uid2inters[uid] = np.array(inters)

    return uid2inters


def preprocess_inters(inters, window_idx):
    (
        qids,
        parts,
        is_corrects,
        is_on_times,
        elapsed_times,
        lag_times,
    ) = inters
    elapsed_times = [min(et / Const.MAX_ELAPSED_TIME_IN_S, 1) for et in elapsed_times]
    lag_times = [min(lt / Const.MAX_LAG_TIME_IN_S, 1) for lt in lag_times]

    ednet_features = {
        "qid": qids,
        "part": parts,
        "is_correct": is_corrects,
        "is_on_time": is_on_times,
        "elapsed_time": elapsed_times,
        "lag_time": lag_times,
    }

    # get all configured features
    start_idx = window_idx
    end_idx = start_idx + ARGS.max_seq_size - 1  # -1 for cls
    all_features = {
        name: np.array(ednet_features[name][start_idx:end_idx])
        for name in ARGS.all_features
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
