import csv
import h5py
import pickle

from config import *


def get_q_info_dic(q_info_path):
    q_info_dic = {}
    questions = h5py.File(q_info_path, "r")["questions"]
    for idx, question in enumerate(questions, start=1):
        q_info_dic[question["question_id"]] = idx

    Const.FEATURE_SIZE["qid"] = len(q_info_dic)
    Const.CLS_VAL["qid"] = len(q_info_dic) + 1
    Const.MASK_VAL["qid"] = len(q_info_dic) + 2

    return q_info_dic


def get_user_interactions_dic(user_inters_base_path):
    with open(user_inters_base_path, "rb") as f_r:
        return pickle.load(f_r)
