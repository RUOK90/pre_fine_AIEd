# pylint: disable=missing-function-docstring,fixme
# Cannot possibly add docstring for all util functions here.
# TODO: continuosly update docstrings here
"""
Utils for both AM pre-training and fine-tuning
"""

import copy
import csv
import datetime

import pickle
import torch

from am_v2 import config


def get_dict_start_time_index():
    dict_start_time_index = {}
    value = 1
    for month in range(1, 13):
        month = str(month)
        month = month if len(month) >= 2 else "0" + month
        for day in range(1, 32):
            day = str(day)
            day = day if len(day) >= 2 else "0" + day
            for hour in range(0, 24):
                hour = str(hour)
                hour = hour if len(hour) >= 2 else "0" + hour
                key = f"{month}_{day}_{hour}"
                dict_start_time_index[key] = value
                value += 1
    return dict_start_time_index


def get_start_time_range():
    start_time_dict = get_dict_start_time_index()

    return (min(start_time_dict.values()), max(start_time_dict.values()))


def write(path, target, mode):
    with open(path, mode) as file_writer:
        file_writer.write(target + "\n")


def transfer_torch(target):
    return torch.Tensor(target).to(config.ARGS.device)


def convert_date(target):
    target = target.split("+")[0]
    target = target.split("Z")[0].split("T")

    v_date = list(map(int, target[0].split("-")))
    v_time = list(map(int, target[1].split(".")[0].split(":")))
    v_sec = list(map(int, target[1].split(".")[1].split(".")))

    return datetime.datetime(
        v_date[0], v_date[1], v_date[2], v_time[0], v_time[1], v_time[2], v_sec[0]
    )


def read_pickle(target_path):
    with open(target_path, "rb") as pkl_file:
        target_file = pickle.load(pkl_file)
    return target_file


def read_mapping_item_id(target_path):
    old2new = {}
    with open(target_path, "r") as mapping_file:
        mapping_list = mapping_file.readlines()
        for line in mapping_list:
            mapping_val = line.replace("\n", "").split(",")
            old2new[int(mapping_val[0])] = int(mapping_val[1])
    return old2new


def get_parameter(*models):
    parameter_list = []
    for cur_model in models:
        parameter_list += list(cur_model.parameters())

    return parameter_list


def convert_list_to_dict(target_list, start_idx):
    cur_id = start_idx
    old_to_new = {}
    new_to_old = {}

    for elements in target_list:
        old_to_new[elements] = cur_id
        new_to_old[cur_id] = elements
        cur_id += 1

    return old_to_new, new_to_old


def get_acc(true_label, predicted_label):
    output2binary = (predicted_label > 0.5).float()
    correct = (output2binary == true_label).float().sum()

    return correct / output2binary.size(0)


def load_weight(to_model, from_model, weight_list):
    for weight in weight_list:
        run = f"{to_model}.{weight}.data={from_model}['{weight}']"
        exec(run)


def load_csv(target_path):
    with open(target_path, "r") as csv_file:
        line_list = csv_file.readlines()
    return line_list


def read_csv(target_path, skip_head=False):
    rows = []
    with open(target_path, newline="") as csv_file:
        spam_reader = csv.reader(csv_file, delimiter=",")
        if skip_head is True:
            next(spam_reader, None)
        for row in spam_reader:
            rows.append(row)
    return rows


def read_paid_at_by_user_id(path):
    paid_at_by_user_id = {}
    rows = read_csv(path, True)
    for user_id, timestring in rows:
        time = string_to_timestamp(timestring)
        paid_at_by_user_id[user_id] = time
    return paid_at_by_user_id


def clones(module, num_layers):
    "Produce num_layers identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


def load_pretrained_weight(weight_path, model, device):
    weight = torch.load(weight_path, map_location=device)
    print(f"Loading {weight_path}")
    # print(weight.keys())
    for name, parm in model.named_parameters():
        if name in weight.keys():
            # print(name, weight[name].shape)
            parm.data.copy_(weight[name])
        # elif name == "embed_time.weight":
        #     print(name, weight["embed_finished_time.weight"].shape)
        #     parm.data.copy_(weight["embed_finished_time.weight"])


def string_to_timestamp(time_string):
    return datetime.datetime.strptime(time_string.split(".")[0], "%Y-%m-%d %H:%M:%S")
