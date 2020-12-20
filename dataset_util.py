import csv


def get_q_info_dic(q_info_path):
    q_info_dic = {}
    with open(q_info_path, "r") as f_r:
        rows = [row for row in csv.DictReader(f_r)]
        for idx, row in enumerate(rows, start=1):
            row["dense_question_id"] = idx
            q_info_dic[int(row["question_id"])] = row
    return q_info_dic
