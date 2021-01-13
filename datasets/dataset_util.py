import csv

from config import *


def get_q_info_dic(q_info_path):
    q_info_dic = {}
    with open(q_info_path, "r") as f_r:
        rows = [row for row in csv.DictReader(f_r)]
        for idx, row in enumerate(rows, start=1):
            row["dense_question_id"] = idx
            q_info_dic[int(row["question_id"])] = row

    Const.FEATURE_SIZE["qid"] = len(q_info_dic)
    Const.CLS_VAL["qid"] = len(q_info_dic) + 1
    Const.MASK_VAL["qid"] = len(q_info_dic) + 2

    return q_info_dic
