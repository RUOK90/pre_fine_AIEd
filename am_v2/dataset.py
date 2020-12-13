# pylint: disable=invalid-name,redefined-outer-name,missing-function-docstring,fixme
# Cannot possibly add docstring for all util functions here.
#
# TODO: change particularly long function names
# TODO: Continuosly update docstrings here
# TODO: Move the outer dictionaries as Dataset attributes

"""
Defines torch Dataset.
"""

import os
import dill as pkl
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from am_v2 import config, constant


def load_sequences(
        data_path,
        users,
        content_mapping,
        mode="train",
):
    """Load sequences into dictionary.
    Loads cached .pkl (using dill) files if exists

    Args:
        data_path: location of the .csv file that contains all user sequences
        users: set of uids (mapped from pk) and window indices
        content_mapping: qid mapping
        mode: train or dev

    Returns:
        uid2sequences: dictionary mapping uid to np array of sequences
            Dimensions of (4, seq_len), where,
            - qids
            - parts
            - is_corrects
            - is_on_times
            are included in such order
    """
    save_path = f"am_v2/load/{mode}_sequences.pkl"
    # if already cached, load cached pkl file
    if os.path.exists(save_path):
        print("Loading pre-processed sequence file:", save_path)
        with open(save_path, "rb") as input_file:
            uid2sequences = pkl.load(input_file)
    else:
        sequences = pd.read_csv(data_path)
        sequences.content_id = sequences.content_id.map(content_mapping)
        print("Pre-processing sequences")
        bool_index_map = {True: constant.TRUE_INDEX, False: constant.FALSE_INDEX}
        uid2sequences = {}
        grouped_by_users = sequences[sequences.student_id.isin(users)].groupby(
            "student_id"
        )
        for user in tqdm(grouped_by_users):
            uid = user[0]
            qids = user[1].content_id.values.tolist()
            parts = user[1].part.values.tolist()
            is_corrects = (
                (user[1].correct_answer == user[1].user_answer)
                    .map(bool_index_map)
                    .values.tolist()
            )
            is_on_times = (
                (user[1].elapsed_time_in_ms <= user[1].time_limit_in_ms)
                    .map(bool_index_map)
                    .values.tolist()
            )
            # elapsed_times in seconds
            elapsed_times = (user[1].elapsed_time_in_ms.values / 1000)
            start_times = user[1].start_time.values
            last_times = np.insert(start_times[:-1], 0, 0)
            lag_times = (start_times - last_times) / 1000 - elapsed_times
            lag_times[0] = 0
            elapsed_times, lag_times = elapsed_times.tolist(), lag_times.tolist()
            uid2sequences[uid] = (
                qids,
                parts,
                is_corrects,
                is_on_times,
                elapsed_times,
                lag_times,
            )

        print("Savng pre-processed sequence file:", save_path)
        with open(save_path, "wb") as output_file:
            pkl.dump(uid2sequences, output_file)

    return uid2sequences


def prep_sequence(sequence, window_start, seq_size):
    """Preprocess a given sequence as such:
        - Retrieve only the selected window of sequences
        - Pad each to max sequence length
        - Mask based on mask ratio

    Args:
        sequence: list of qids, parts, is_corrects, and is_on_times
        window_start: index of window start
        seq_size: max seq length

    Returns:
        qids: list of content ids
        parts: list of parts
        is_predicted_list: Boolean list of predicted interactions (inverted mask)
        input_correctness: the masked input response correctness
        input_timeliness: the masked input timeliness
        output_correctness: the masked output response correctness
        output_timeliness: the masked output timeliness
    """
    max_elapsed_time = 300
    max_lag_time = 86400
    qids, parts, is_corrects, is_on_times, elapsed_times, lag_times, = sequence
    elapsed_times = [min(et/max_elapsed_time, 1) for et in elapsed_times]
    lag_times = [min(lt/max_lag_time, 1) for lt in lag_times]
    features = {
        'qid': qids,
        'part': parts,
        'is_correct': is_corrects,
        'is_on_time': is_on_times,
        'elapsed_time': elapsed_times,
        'lag_time': lag_times
    }

    start_index = window_start
    end_index = start_index + seq_size
    # Retrieve selected window for each sequence

    features = {name: features[name][start_index:end_index] for name in features}

    # create random mask
    mask_p = config.ARGS.mask_random_ratio
    # Leaves only interactions that are kept as inputs (40% are True)
    mask = np.random.random_sample(len(features['qid'])) > mask_p
    inverted_mask = ~mask
    features_to_mask = {
        'is_correct': (np.array(features['is_correct']), bool),
        'is_on_time': (np.array(features['is_correct']), bool),
        'elapsed_time': (np.array(features['elapsed_time']), float),
        'lag_time': (np.array(features['elapsed_time']), float),
    }
    masked_features = mask_sequences(inverted_mask, features_to_mask)
    features['is_predicted'] = inverted_mask.tolist()
    features['is_correct'] = (2 - np.array(features['is_correct'])).tolist()
    features['is_on_time'] = (2 - np.array(features['is_on_time'])).tolist()

    feature_dict = {'all': features, 'masked': masked_features}
    # create and zero padding
    zero_count = max(seq_size - len(features['qid']), 0)
    zero_padding = [constant.PAD_INDEX] * zero_count
    feature_dict = {
        feature_group: {
            name: torch.Tensor(feature + zero_padding)
            for name, feature in features.items()
        } for feature_group, features in feature_dict.items()
    }
    return feature_dict


def mask_sequences(inverted_mask, input_features):
    """Mask a given sequence randomly

    TODO: 해당 코드에서 test때는 마지막것만 masking하도록 해야함

    Args:
        features: list of features
            - is_corrects: full response correctness sequence
            - is_on_times: full timeliness sequence
            - elapsed_times
            - tags
            - ...
        feature_types: list of feature types
            - Boolean (e.g. is_corrects)
            - Integer (e.g. tags)
            - Floats (e.g. elapsed_times)

    Returns:
        inverted_mask,
        nput_correctness,
        input_timeliness,
        output_correctness,
        output_timeliness,
    """
    masked_features = {}
    for key, input_feature in input_features.items():
        feature, feature_type = input_feature

        # output_feature = np.array(feature).copy()
        if feature_type is bool:
            feature[inverted_mask] = constant.IS_CORRECT_MASK_INDEX
            # use 1/0 indexing instead of 1/2 for T/F
            # output_feature = -(output_feature - 2)
        elif feature_type is float:
            feature[inverted_mask] = constant.FLOAT_TASK_MASK_INDEX
        elif feature_type is int:
            feature[inverted_mask] = constant.INT_TASK_MASK_INDEX
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        masked_features[key] = feature.tolist()
    # output_feature[mask] = constant.PAD_INDEX

    return masked_features


class DataSet(torch.utils.data.Dataset):
    """
    Pytorch Dataset class that dynamically reads from data tree
    """

    def __init__(
            self,
            name,
            sample_list,
            content_mapping,
            sequence_size,
            is_training,
    ):
        self._name = name
        self._sample_list = sample_list
        self._content_mapping = content_mapping
        self._sequence_size = sequence_size
        self._is_training = is_training
        self._sequences = self.load_all_data(config.ARGS.data_file)

    def load_all_data(self, data_file):
        """Load all user data into memory

        Args:
            data_file: which data to use (e.g. new_am_data.csv)

        Returns:
            sequences: list of all sequences preprocessed and masked
        """
        uids = set([uid for (uid, _) in self._sample_list])
        sequences = load_sequences(data_file, uids, self._content_mapping, self._name)
        return sequences

    @property
    def the_number_of_samples(self):
        return len(self._sample_list)

    def __repr__(self):
        return f"{self._name}: # of samples: {self.the_number_of_samples}"

    def __len__(self):
        return len(self._sample_list)

    def __getitem__(self, index):
        uid, window_index = self._sample_list[index]
        feature_dict = prep_sequence(
            self._sequences[uid], window_index, self._sequence_size
        )

        return feature_dict


def make_dataloader(dataset, batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
