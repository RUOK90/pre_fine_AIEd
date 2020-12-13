import torch
from torch.utils import data
from am_v2 import config

def get_user_data(base_path, old_to_new_item_id, user_id_list):
    """
    Load user / review_correctness / KT data
    """

    uid2sequences = {}
    uid2review_qid = {}
    uid2is_review_correct = {}
    new_to_old_item_id = {
        new: old
        for old, new in old_to_new_item_id.items()
    }
    for user_id in user_id_list:
        user_file_path = f'{base_path}/{user_id}'

        uid2review_qid[user_id] = int(user_id.split("_")[1])
        uid2is_review_correct[user_id] = 2 - int(user_id.split("_")[-1].split('.')[0])

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

        for line in line_list[1:]:
            line = line.split(",")
            # if int(line[0]) not in old_to_new_item_id:
            #    continue
            if config.ARGS.load_question_embed in ['bert', 'quesnet', 'word2vec']:
                qid = new_to_old_item_id[int(line[0])]
            else:
                qid = int(line[0])
            part = int(line[1])
            is_correct = int(line[2])
            is_on_time = int(line[3])
            elapsed_time = min(float(line[4])/max_elapsed_time, 1)
            lag_time = min(float(line[5])/max_lag_time, 1)

            qid_list.append(qid)
            part_list.append(part)
            is_correct_list.append(is_correct)
            is_on_time_list.append(is_on_time)
            elapsed_time_list.append(elapsed_time)
            lag_time_list.append(lag_time)
        
        uid2sequences[user_id] = {
            'qid': qid_list,
            'part': part_list,
            'is_correct': is_correct_list,
            'is_on_time': is_on_time_list,
            'elapsed_time': elapsed_time_list,
            'lag_time': lag_time_list
        }
        # if config.ARGS.debug_mode:
        # print(uid2review_qid[user_id], uid2is_review_correct[user_id])
        # print(uid2sequences[user_id])
    return uid2sequences, uid2review_qid, uid2is_review_correct


class DataSet(data.Dataset):
    """
    DataSet class for review prediction data

    Args:
        is_training: training mode
        base_path: base path for data loading
        content_mapping: sequential mapping for questions
        user_id_list: list of user ids
        sequence_size: maximum size of sequence
    """

    def __init__(
        self,
        is_training,
        base_path,
        content_mapping,
        user_id_list,
        sequence_size,
    ):
        self._is_training = is_training
        self._user_id_list = user_id_list
        self._sequences, self._review_qid, self._is_review_correct = get_user_data(
            base_path, content_mapping, user_id_list
        )

        self._sequence_size = sequence_size

    @property
    def the_number_of_users(self):
        """
        Size of dataset in terms of users
        """
        return len(self._user_id_list)

    def _get_x(self, user_id):
        """
        Gets single input for batching
        During training, use augmented input (random samplin)
        During testing, use last few samples
        """
        input_features = self._sequences[user_id]

        num_pads = max(0, self._sequence_size - len(input_features['qid']))
        zero_pads = [0] * num_pads
        features = {
            name: torch.Tensor(
                feature[-self._sequence_size:] + zero_pads
            )
            for name, feature in input_features.items()
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
        review_qid = self._review_qid[user_id]
        is_review_correct = self._is_review_correct[user_id]
        return (
            features,
            review_qid,
            is_review_correct
        )

