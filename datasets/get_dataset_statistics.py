import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


debug = False

if debug:
    with open(
        "/private/datasets/magneto_2021-01-27/user_interactions_wo_lecture_debug.pkl",
        "rb",
    ) as f_r:
        user_interactions = pickle.load(f_r)
else:
    with open(
        "/private/datasets/magneto_2021-01-27/user_interactions_wo_lecture.pkl", "rb"
    ) as f_r:
        user_interactions = pickle.load(f_r)

inter_len_cnt_dic = defaultdict(int)
inter_len_list = []
elapsed_time_list = []
exp_time_list = []
lag_time_list = []
user_cnt = 0
correct_cnt = 0
incorrect_cnt = 0
on_time_cnt = 0
not_on_time_cnt = 0
for i, (user, interactions) in enumerate(tqdm(user_interactions.items())):
    if debug and i == 500:
        break
    if len(interactions) >= 15:
        user_cnt += 1
        inter_len_cnt_dic[len(interactions)] += 1
        inter_len_list.append(len(interactions))

        for inter in interactions:
            if inter["is_correct"] == 0:
                incorrect_cnt += 1
            elif inter["is_correct"] == 1:
                correct_cnt += 1

            if inter["timeliness"] == 0:
                not_on_time_cnt += 1
            elif inter["timeliness"] == 1:
                on_time_cnt += 1

            elapsed_time_list.append(inter["elapsed_time_in_s"])
            exp_time_list.append(inter["exp_time_in_s"])
            lag_time_list.append(inter["lag_time_in_s"])

inter_len_cnt_list = list(inter_len_cnt_dic.items())
inter_len_cnt_list.sort()
inter_len_list.sort()

print("# of students", user_cnt)
print("# of inters", np.sum(inter_len_list))
print("min_inter_len", np.min(inter_len_list))
print("max_inter_len", np.max(inter_len_list))
print("mean_inter_len", np.mean(inter_len_list))
print("median_inter_len", np.median(inter_len_list))
print("correct_response_ratio", correct_cnt / (correct_cnt + incorrect_cnt))
print("on_time_response_ratio", on_time_cnt / (on_time_cnt + not_on_time_cnt))

# eid_cnt_list = list(eid_cnt_dic.values())
# # eid_cnt_list.sort(reverse=True)
# bins = [i for i in range(500)]
# plt.hist(eid_cnt_list, bins=bins)
# plt.tick_params(
#     axis="x",  # changes apply to the x-axis
#     which="both",  # both major and minor ticks are affected
#     bottom=False,  # ticks along the bottom edge are off
#     top=False,  # ticks along the top edge are off
#     labelbottom=False,
# )  # labels along the bottom edge are off
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.xlabel("Exercise IDs")
# plt.ylabel("Counts")
# plt.show()

# elapsed_time_list.sort()
bins = [i for i in range(150)]
plt.hist(elapsed_time_list, bins=bins)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.xlabel("Elapsed Times (Seconds)")
plt.ylabel("Counts")
plt.show()

# exp_time_list.sort()
bins = [i for i in range(150)]
plt.hist(exp_time_list, bins=bins)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.xlabel("Explanation Times (Seconds)")
plt.ylabel("Counts")
plt.show()

# lag_time_list.sort()
bins = [i for i in range(150)]
plt.hist(lag_time_list, bins=bins)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.xlabel("Inactive Times (Seconds)")
plt.ylabel("Counts")
plt.show()

# with open("/private/datasets/magneto_2021-01-27/user_score_idxs.pkl", "rb") as f_r:
#     user_score_idxs = pickle.load(f_r)
#
# user_cnt_dic = defaultdict(int)
# inter_len_list = []
# score_list = []
# user_score_idxs = (
#     user_score_idxs[0]["train"] + user_score_idxs[0]["val"] + user_score_idxs[0]["test"]
# )
#
# for uid, end_idx, lc, rc in user_score_idxs:
#     user_cnt_dic[uid] += 1
#     inter_len_list.append(end_idx)
#     score_list.append(lc + rc)
#
# print("# of students", len(user_cnt_dic))
# print("# of scores", len(user_score_idxs))
# print("min_inter_len", np.min(inter_len_list))
# print("max_inter_len", np.max(inter_len_list))
# print("mean_inter_len", np.mean(inter_len_list))
# print("median_inter_len", np.median(inter_len_list))
#
# score_list.sort()
#
# bins = range(0, 1000, 20)
# plt.hist(score_list, bins=bins)
# plt.xlabel("Scores")
# plt.ylabel("Counts")
# # plt.title("Score Distribution")
# # plt.grid(True)
# plt.show()
