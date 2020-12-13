"""Util functions to read data
"""
import os

import dill as pkl
import pandas as pd

from tqdm import tqdm
import numpy as np


def create_parent_path(target_path, file_name):
    """Helper func to create parent path for a data tree file"""
    first, second, third, fourth = name_to_keys(file_name)
    return f"{target_path}/{first}/{second}/{third}/{fourth}"


def name_to_keys(name):
    """Helper func get first four paths of a data tree file"""
    # return [name[index] for index in range(0, 4)]
    first = name[0]
    second = name[1]
    third = name[2]
    fourth = name[3]
    return first, second, third, fourth


def create_full_path(target_path, file_name):
    """Helper func to create full path for a data tree file"""
    return f"{create_parent_path(target_path, file_name)}/{file_name}"


def get_sample_list_from_csv(data_path, target_csv, min_size):
    """Helper func to get window indices and user lists from csv file"""
    num_of_users = 0
    sample_list = []

    start_idx = 0

    with open(target_csv, "r") as file:
        line_list = file.readlines()
        for i in range(start_idx, len(line_list)):
            cur_user = line_list[i].replace("\n", "")
            cur_user_data_path = create_full_path(data_path, cur_user)
            with open(cur_user_data_path, "r") as csv_file:
                num_of_samples = len(csv_file.readlines())
            if num_of_samples <= min_size:
                continue
            num_of_users += 1

            sample_list += [[cur_user, index] for index in range(num_of_samples)]

    return sample_list, num_of_users


def save_pkl(data, path):
    """Save Pickle file"""
    with open(path, "wb") as output_file:
        pkl.dump(data, output_file)
    return 1


def load_pkl(path):
    """Load Pickle file"""
    with open(path, "rb") as pkl_file:
        data = pkl.load(pkl_file)
    return data


def get_user_windows(
    data_path, min_size, use_old_split=False, seq_size=100, stride=50,
):
    """Helper func to get window indices and user list from new data format"""
    train_users, dev_users, test_users = load_pkl("am_v2/load/ednet_user_split.pkl")
    save_path = f"am_v2/load/ednet_sample_list_{stride}.pkl"
    if os.path.exists(save_path):
        print(f"Loading user window splits from {save_path}")
        train_sample_list, dev_sample_list, test_sample_list = load_pkl(save_path)
    else:
        print(f"Creating user window splits from {save_path}")
        sequences = pd.read_csv(data_path)

        train_groups = sequences[sequences.student_id.isin(set(train_users))].groupby(
            "student_id"
        )
        dev_groups = sequences[sequences.student_id.isin(set(dev_users))].groupby(
            "student_id"
        )
        test_groups = sequences[sequences.student_id.isin(set(test_users))].groupby(
            "student_id"
        )

        train_sample_list, dev_sample_list, test_sample_list = [], [], []
        for user in tqdm(train_groups):
            uid = user[0]
            num_samples = len(user[1])
            if num_samples <= min_size:
                continue
            range_end = max(num_samples - seq_size + stride, 1)
            train_sample_list += [(uid, i) for i in range(0, range_end, stride)]
        for user in tqdm(dev_groups):
            uid = user[0]
            num_samples = len(user[1])
            if num_samples <= min_size:
                continue
            range_end = max(num_samples - seq_size + stride, 1)
            dev_sample_list += [(uid, i) for i in range(0, range_end, stride)]
        for user in tqdm(test_groups):
            uid = user[0]
            num_samples = len(user[1])
            if num_samples <= min_size:
                continue
            range_end = max(num_samples - seq_size + stride, 1)
            test_sample_list += [(uid, i) for i in range(0, range_end, stride)]
        save_pkl([train_sample_list, dev_sample_list, test_sample_list], save_path)

    return (
        train_sample_list,
        dev_sample_list,
        test_sample_list,
    )
