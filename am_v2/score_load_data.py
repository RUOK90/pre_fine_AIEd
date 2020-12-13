"""
Data and model loading util functions
"""
import csv
import datetime

from am_v2 import config, constant


def get_time_limit_in_ms(question_id, time_limits):
    """
    Load time limit data
    """
    return float(time_limits.get(str(question_id), constant.DEFAULT_TIME_LIMIT_IN_MS))


def get_user_data(base_path, old_to_new_item_id, user_id_list):
    """
    Load user / score / KT data
    """
    # Load time limit data from csv
    time_limits = {}
    with open("am_v2/tmp/time_limits.csv", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            question_id = row["question_id"]
            time_limit_in_ms = row["time_limit_in_ms"]
            time_limits[question_id] = time_limit_in_ms

    uid2sequences = {}
    uid2scores = {}
    for user_id in user_id_list:
        user_file_path = f'{base_path}/{user_id}'

        lc_score = int(user_id.split("_")[1])
        rc_score = int(user_id.split("_")[2])

        uid2scores[user_id] = {'lc': lc_score, 'rc': rc_score}
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
        time_format = "%Y-%m-%d %H:%M:%S"
        last_time = None
        for line in line_list:
            line = line.split(",")
            if int(line[1]) not in old_to_new_item_id:
                continue

            if config.ARGS.load_question_embed in ['bert', 'quesnet', 'word2vec']:
                qid = int(line[1])
            else:
                qid = old_to_new_item_id[int(line[1])]
            is_correct = constant.TRUE_INDEX if line[2] == line[3] else constant.FALSE_INDEX
            if line[constant.PART_INDEX] == "unknown":
                part = 8
            else:
                part = int(line[constant.PART_INDEX])
            elapsed_time_in_ms = float(line[5])
            time_limit_in_ms = get_time_limit_in_ms(int(line[1]), time_limits)
            is_on_time = (
                constant.TRUE_INDEX
                if elapsed_time_in_ms <= time_limit_in_ms
                else constant.FALSE_INDEX
            )
            elapsed_time = elapsed_time_in_ms / 1000
            curr_time = datetime.datetime.strptime(line[0].split('.')[0], time_format)
            if last_time is None:
                lag_time = 0
            else:
                lag_time = (curr_time - last_time).total_seconds() - elapsed_time
            last_time = curr_time

            elapsed_time = min(elapsed_time / max_elapsed_time, 1)
            lag_time = min(lag_time / max_lag_time, 1)
            if config.ARGS.use_buggy_timeliness:
                is_on_time = constant.TRUE_INDEX

            qid_list.append(qid)
            part_list.append(part)
            is_correct_list.append(is_correct)
            is_on_time_list.append(is_on_time)
            elapsed_time_list.append(elapsed_time)
            lag_time_list.append(lag_time)

        uid2sequences[user_id]={
            'qid': qid_list,
            'part': part_list,
            'is_correct': is_correct_list,
            'is_on_time': is_on_time_list,
            'elapsed_time': elapsed_time_list,
            'lag_time': lag_time_list
        }

    return uid2sequences, uid2scores
