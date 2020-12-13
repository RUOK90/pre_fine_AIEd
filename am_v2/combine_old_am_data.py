"""
Combine old data tree format to single csv file
sorted by user_id and start_time
"""
import time

import pandas as pd

from tqdm import tqdm

from am_v2 import read_data


if __name__ == "__main__":
    BASE_PATH = "/shared/modified_AAAI20/response"
    DATA_PATH = f"{BASE_PATH}/data_tree"

    train_users = set(
        pd.read_csv(f"{BASE_PATH}/train_user_list.csv", header=None)[0].values
    )
    val_users = set(
        pd.read_csv(f"{BASE_PATH}/val_user_list.csv", header=None)[0].values
    )
    test_users = set(
        pd.read_csv(f"{BASE_PATH}/test_user_list.csv", header=None)[0].values
    )
    users = train_users | val_users | test_users

    uid_pk_admin = pd.read_csv(f"/shared/new_am_data/students_final_ver2.csv")
    pk2id = uid_pk_admin.set_index("pk").id.to_dict()

    user_csvs = []
    for user_file in tqdm(users):
        pk = user_file.split(".")[0]
        user_csv = pd.read_csv(
            read_data.create_full_path(DATA_PATH, user_file),
            header=None,
            names=[
                "start_time",
                "content_id",
                "user_answer",
                "correct_answer",
                "part",
                "elapsed_time_in_ms",
            ],
            usecols=[0, 1, 2, 3, 4, 5],
        )
        user_csv.insert(loc=0, column="student_id", value=[pk2id[pk]] * len(user_csv))
        user_csvs.append(user_csv)

    print(f"read {len(user_csvs)} files")
    user_csvs = pd.concat(user_csvs).reset_index(drop=True)

    print(f"Cast parts to ints and assign unknowns 8")
    user_csvs.loc[user_csvs.part == "unknown", "part"] = 8
    user_csvs.part = user_csvs.part.astype(int)

    print("fill unknowns using time_limit.csv")
    time_limits = pd.read_csv("am_v2/tmp/time_limits.csv")
    qid2part = time_limits.set_index("question_id").part_number.astype(int).to_dict()
    user_csvs.loc[user_csvs.part == 8, "part"] = user_csvs[
        user_csvs.part == 8
    ].content_id.map(qid2part)
    print("Add time_limits_in_ms column using time_limit.csv")
    qid2limit = (
        time_limits.set_index("question_id").time_limit_in_ms.astype(int).to_dict()
    )
    user_csvs["time_limit_in_ms"] = user_csvs.content_id.map(qid2limit)

    print("saving as '/shared/new_am_data/old_am_data.csv'")
    start_time = time.time()
    user_csvs.to_csv("/shared/new_am_data/old_am_data.csv", index=False)
    print(f"Done saving: {time.time() - start_time} secs")
