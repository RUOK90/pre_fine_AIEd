import h5py
import json
import numpy as np
from tqdm import tqdm
import pickle
from datetime import datetime
import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt


random.seed(1234)
data_dir = "/private/datasets/magneto_2021-01-27"
interactions_file = "interactions.hdf5"
user_file = "users.hdf5"
score_interactions_file = "score_interactions.hdf5"
score_user_file = "score_users.hdf5"
scores_file = "scores.json"

interactions = h5py.File(f"{data_dir}/{interactions_file}", "r")
score_interactions = h5py.File(f"{data_dir}/{score_interactions_file}", "r")
users = h5py.File(f"{data_dir}/{user_file}", "r")
score_users = h5py.File(f"{data_dir}/{score_user_file}", "r")
with open(f"{data_dir}/{scores_file}", "r") as f_r:
    scores = json.load(f_r)

# score duplication check
uid_dic = {}
for uid, e in scores.items():
    assert uid not in uid_dic
    uid_dic[uid] = None
    date_dic = {}
    for date in e.keys():
        assert date not in date_dic
        date_dic[date] = None

# filtered columns
columns = [
    "submitted_at",
    "item_id",
    "part_id",
    "user_answer",
    "elapsed_time_in_s",
    "is_correct",
    "timeliness",
    "exp_time_in_s",
    "lag_time_in_s",
]


# remove lecture watching interactions and save interactions dictionary to npy format
def save_filtered_user_interactions_dic(users, interactions, min_seq_len, file_name):
    filtered_user_interactions_dic = {}
    for i, user in enumerate(tqdm(users)):
        if debug and i == 1000:
            break
        user_id = user["user_id"]
        start_idx = user["user_interaction_index_start"]
        end_idx = user["user_interaction_index_end"]
        user_interactions = interactions[start_idx : end_idx + 1]
        mask = np.array(
            [
                True if inter["content_type_abrg"] == 1 else False
                for inter in user_interactions
            ]
        )
        if len(user_interactions[mask][columns]) >= min_seq_len:
            filtered_user_interactions_dic[user_id] = user_interactions[mask][columns]

    if debug:
        file_name += "_debug"
    with open(f"{data_dir}/{file_name}.pkl", "wb") as f_w:
        pickle.dump(filtered_user_interactions_dic, f_w, pickle.HIGHEST_PROTOCOL)


def get_filtered_user_interactions_dic(file_name):
    if debug:
        file_name += "_debug"
    with open(f"{data_dir}/{file_name}.pkl", "rb") as f_r:
        return pickle.load(f_r)


def check_user_interactions(user_interactions_dic):
    len_cnt_dic = defaultdict(int)
    for user, interactions in tqdm(user_interactions_dic.items()):
        len_cnt_dic[len(interactions)] += 1

    len_cnt_list = list(len_cnt_dic.items())
    len_cnt_list.sort()

    x, y = zip(*len_cnt_list)
    plt.figure()
    plt.bar(x[:100], y[:100])
    plt.xlabel("seq len")
    plt.ylabel("cnt")
    plt.savefig("score_len_cnt.png")
    pass


def save_score_idxs_dic(user_interactions_dic, scores, min_seq_len, n_folds, file_name):
    user_end_idx_lc_rc_dic = {}
    for user, date_lc_rc_dic in tqdm(scores.items()):
        user = int(user)
        if user in user_interactions_dic:
            date_lc_rc_list = [
                [
                    datetime.strptime(date, "%Y.%m.%d").timestamp(),
                    lc_rc["lc"],
                    lc_rc["rc"],
                ]
                for date, lc_rc in date_lc_rc_dic.items()
                if datetime.strptime(date, "%Y.%m.%d").timestamp()
                > user_interactions_dic[user][0][0]
            ]
            date_lc_rc_list.sort()

            date_idx = 0
            inter_idx = 0
            prev_inter_idx = -1
            end_idx_lc_rc = []
            user_interactions = user_interactions_dic[user]
            while inter_idx < len(user_interactions) and date_idx < len(
                date_lc_rc_list
            ):
                if date_lc_rc_list[date_idx][0] < user_interactions[inter_idx][0]:
                    lc = date_lc_rc_list[date_idx][1]
                    rc = date_lc_rc_list[date_idx][2]
                    # if inter_idx >= min_seq_len:
                    #     if prev_inter_idx == inter_idx:
                    #         end_idx_lc_rc.pop()
                    #     end_idx_lc_rc.append((inter_idx, lc, rc))
                    #     prev_inter_idx = inter_idx

                    if inter_idx >= min_seq_len:
                        correct_ratio = np.mean(
                            user_interactions[:inter_idx]["is_correct"]
                        )
                        if lc + rc >= 200 and correct_ratio >= 0.1:
                            # end_idx_lc_rc.append((inter_idx, lc, rc, correct_ratio))
                            end_idx_lc_rc.append((inter_idx, lc, rc))

                    date_idx += 1
                    continue
                inter_idx += 1

            if len(end_idx_lc_rc) != 0:
                user_end_idx_lc_rc_dic[user] = end_idx_lc_rc

    # # plotting
    # len_cnt_dic = defaultdict(int)
    # cr_score_list = []
    # for user, end_idx_lc_rc in user_end_idx_lc_rc_dic.items():
    #     for end_idx, lc, rc, cr in end_idx_lc_rc:
    #         if lc + rc <= 200 or lc + rc > 990 or cr <= 0.1:
    #             print(user, end_idx, lc + rc, cr)
    #         len_cnt_dic[end_idx] += 1
    #         cr_score_list.append((cr, lc + rc))
    # len_cnt_list = list(len_cnt_dic.items())
    # len_cnt_list.sort()

    # # plot len-cnt
    # x, y = zip(*len_cnt_list)
    # plt.figure()
    # plt.bar(x[:2000], y[:2000])
    # plt.xlabel("seq len")
    # plt.ylabel("cnt")
    # plt.show()

    # # plot cr-score
    # x, y = zip(*cr_score_list)
    # plt.figure()
    # plt.scatter(x, y)
    # plt.xlabel("correctness ratio")
    # plt.ylabel("score")
    # plt.show()

    # split user_end_idx_lc_rc_dic by n_folds and get train, val, test set
    user_score_idxs = []
    user_end_idx_lc_rc_list = list(user_end_idx_lc_rc_dic.items())
    random.shuffle(user_end_idx_lc_rc_list)
    n_users_per_fold = math.ceil(len(user_end_idx_lc_rc_list) / n_folds)
    users_per_fold = [
        user_end_idx_lc_rc_list[i * n_users_per_fold : (i + 1) * n_users_per_fold]
        for i in range(n_folds)
    ]
    for i in range(n_folds):
        train_user_score_idxs = (
            users_per_fold[i % n_folds]
            + users_per_fold[(i + 1) % n_folds]
            + users_per_fold[(i + 2) % n_folds]
        )
        user_score_idxs_train = []
        for user, score_idxs in train_user_score_idxs:
            for idx, lc, rc in score_idxs:
                user_score_idxs_train.append((user, idx, lc, rc))

        val_user_score_idxs = users_per_fold[(i + 3) % n_folds]
        user_score_idxs_val = []
        for user, score_idxs in val_user_score_idxs:
            for idx, lc, rc in score_idxs:
                user_score_idxs_val.append((user, idx, lc, rc))

        test_user_score_idxs = users_per_fold[(i + 4) % n_folds]
        user_score_idxs_test = []
        for user, score_idxs in test_user_score_idxs:
            for idx, lc, rc in score_idxs:
                user_score_idxs_test.append((user, idx, lc, rc))

        user_score_idxs.append(
            {
                "train": user_score_idxs_train,
                "val": user_score_idxs_val,
                "test": user_score_idxs_test,
            }
        )

    if debug:
        file_name += "_debug"
    with open(f"{data_dir}/{file_name}.pkl", "wb") as f_w:
        pickle.dump(user_score_idxs, f_w, pickle.HIGHEST_PROTOCOL)


def check_score_idxs(file_name):
    if debug:
        file_name += "_debug"
    with open(f"{data_dir}/{file_name}.pkl", "rb") as f_r:
        user_score_idxs = pickle.load(f_r)

    user_score_idxs = (
        user_score_idxs[0]["train"]
        + user_score_idxs[0]["val"]
        + user_score_idxs[0]["test"]
    )

    len_cnt_dic = defaultdict(int)
    for user, end_idx, lc, rc in user_score_idxs:
        if lc + rc <= 200 or lc + rc >= 990:
            print(user, end_idx, lc, rc)
        len_cnt_dic[end_idx] += 1
    len_cnt_list = list(len_cnt_dic.items())
    len_cnt_list.sort()

    x, y = zip(*len_cnt_list)
    plt.figure()
    plt.bar(x[:2000], y[:2000])
    plt.xlabel("seq len")
    plt.ylabel("cnt")
    # plt.show()
    pass


def save_user_interaction_windows_dic(
    user_interactions_dic, scores, min_seq_len, max_seq_len, interval, file_name
):
    user_interaction_windows_train_dic = {}
    user_interaction_windows_val_dic = {}
    for user, interactions in tqdm(user_interactions_dic.items()):
        if len(interactions) < min_seq_len:
            continue
        inter_idx = 0
        interaction_windows = []
        while inter_idx + max_seq_len < len(interactions):
            interaction_windows.append((inter_idx, inter_idx + max_seq_len))
            inter_idx += interval
        interaction_windows.append(
            (max(len(interactions) - max_seq_len, 0), len(interactions))
        )

        if str(user) in scores:
            user_interaction_windows_val_dic[user] = interaction_windows
        else:
            user_interaction_windows_train_dic[user] = interaction_windows

    user_interaction_windows_train_list = list(
        user_interaction_windows_train_dic.items()
    )
    user_interaction_windows_val_list = list(user_interaction_windows_val_dic.items())
    random.shuffle(user_interaction_windows_train_list)
    if not debug:
        user_interaction_windows_val_list += user_interaction_windows_train_list[:10000]
        user_interaction_windows_train_list = user_interaction_windows_train_list[
            10000:
        ]

    user_interaction_windows_train = []
    for user, interaction_windows in user_interaction_windows_train_list:
        for start_idx, end_idx in interaction_windows:
            user_interaction_windows_train.append((user, start_idx, end_idx))

    user_interaction_windows_val = []
    for user, interaction_windows in user_interaction_windows_val_list:
        for start_idx, end_idx in interaction_windows:
            user_interaction_windows_val.append((user, start_idx, end_idx))

    user_interaction_windows = {
        "train": user_interaction_windows_train,
        "val": user_interaction_windows_val,
    }

    file_name += f"_{max_seq_len}_{interval}"
    if debug:
        file_name += "_debug"
    with open(f"{data_dir}/{file_name}.pkl", "wb") as f_w:
        pickle.dump(user_interaction_windows, f_w, pickle.HIGHEST_PROTOCOL)


debug = False

# save_filtered_user_interactions_dic(
#     users["users_integrated"],
#     interactions["interactions_integrated"],
#     10,
#     "user_interactions_wo_lecture",
# )

filtered_user_interactions_dic = get_filtered_user_interactions_dic(
    "user_interactions_wo_lecture"
)
# check_user_interactions(filtered_user_interactions_dic)

save_score_idxs_dic(filtered_user_interactions_dic, scores, 10, 5, "user_score_idxs")
# check_score_idxs("user_score_idxs")

# save_user_interaction_windows_dic(
#     filtered_user_interactions_dic, scores, 15, 1023, 256, "user_interaction_windows"
# )
#
# save_user_interaction_windows_dic(
#     filtered_user_interactions_dic, scores, 15, 8191, 2048, "user_interaction_windows"
# )

# save_user_interaction_windows_dic(
#     filtered_user_interactions_dic, scores, 15, 1023, 128, "user_interaction_windows"
# )
#
# save_user_interaction_windows_dic(
#     filtered_user_interactions_dic, scores, 15, 8191, 1024, "user_interaction_windows"
# )
