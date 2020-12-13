import numpy as np
import pandas as pd
import ast
import torch
from tqdm import tqdm
from am_v2 import config, constant


def load_data(data_path='am_v2/load/texts_with_token_ids.csv'):
    """Load question texts into dictionary.

    Args:
        data_path: location of the .csv file that contains qid to text info

    Returns:
        qid_to_tokens:
        sample_list:
    """
    qid_to_tokens = {}
    text_data = pd.read_csv(data_path)
    sample_list = []
    for _, data in tqdm(text_data.iterrows()):
        choice_tokens = sum(ast.literal_eval(data.choice_token_ids), [])
        passage_tokens = ast.literal_eval(data.passage_token_ids)
        question_tokens = ast.literal_eval(data.question_token_ids)
        all_tokens = (passage_tokens + question_tokens + choice_tokens)
        qid = data.question_id
        qid_to_tokens[qid] = all_tokens
        sample_list += [(qid, i) for i in range(len(all_tokens))]

    return qid_to_tokens, sample_list


class BERTDataSet(torch.utils.data.Dataset):
    """
    Pytorch Dataset class that dynamically reads from data tree
    """

    def __init__(
            self,
            # name,
            num_vocab,
            qid_to_tokens,
            sample_list,
            # content_mapping,
            sequence_size,
            is_training,
    ):
        # self._name = name
        # self._content_mapping = content_mapping
        self._mask_token = num_vocab + 1
        self._sequence_size = sequence_size
        self._is_training = is_training
        self._qid_to_tokens = qid_to_tokens
        self._sample_list = sample_list

    @property
    def the_number_of_samples(self):
        return len(self._sample_list)

    def __repr__(self):
        return f"# of samples: {self.the_number_of_samples}"

    def __len__(self):
        return len(self._sample_list)

    def __getitem__(self, index):
        qid, start_idx = self._sample_list[index]
        tokens = np.array(self._qid_to_tokens[qid][start_idx:start_idx+100])
        mask = np.random.random_sample(len(tokens)) < 0.15
        masked_tokens = np.copy(tokens)
        masked_tokens[mask] = self._mask_token
        num_pads = max(0, 100 - len(tokens))
        tokens = np.concatenate((tokens, np.zeros(num_pads))).tolist()
        masked_tokens = np.concatenate((masked_tokens, np.zeros(num_pads))).tolist()
        is_predicted = np.concatenate((mask, np.zeros(num_pads))).tolist()
        features = {
            'all': {
                'word_token': torch.Tensor(tokens),
                'is_predicted': torch.Tensor(is_predicted),
            },
            'masked': {
                'word_token': torch.Tensor(masked_tokens),
            },
        }
        return features
