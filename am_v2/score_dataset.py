"""
Dataset class for score prediction data
"""
import random

import torch
from torch.utils import data
import numpy as np

from am_v2 import score_load_data


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
        self._sequences, self._scores = score_load_data.get_user_data(
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
