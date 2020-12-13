"""
Script to get AM data for
- Pretraining
- Score modeling
"""

import time

import pandas as pd


if __name__ == "__main__":
    start_time = time.time()
    BASE_PATH = "/shared/new_am_data"

    print(f"Loading TCR data from '{BASE_PATH}/tcr.csv'")
    tcr = pd.read_csv(f"{BASE_PATH}/tcr.csv")
    cur_time = time.time()
    print(f"Done loading TCR data: {cur_time - start_time} sec")

    # Get uid to pk translations
    print(f"Loading uid2pk data from '{BASE_PATH}/students_final_ver2.csv'")
    uid_pk_admin = pd.read_csv(f"{BASE_PATH}/students_final_ver2.csv")
    pk2id = uid_pk_admin.set_index("pk").id.to_dict()

    # Get in and out filters for uids
    admins = set(uid_pk_admin[uid_pk_admin.admin].id.tolist())
    id_in_pos = set(
        [
            pk2id[pk]
            for pk in set(pd.read_csv(f"{BASE_PATH}/user_resource.csv").pk.values)
            if pk in pk2id
        ]
    )
    score_ids = set(
        [
            pk2id[pk]
            for pk in set(pd.read_csv(f"{BASE_PATH}/score_filter.csv").user_id.values)
        ]
    )
    uid_ja = set(pd.read_csv(f"{BASE_PATH}/payment_ja.csv").student_id.tolist())

    in_filters = id_in_pos
    out_filters = admins | score_ids | uid_ja
    cur_time = time.time()
    print(f"Done loading uid2pk data: {cur_time - start_time} sec")

    print("Filtering TCR by content type")
    tcr = tcr[tcr.content_type_abrg == 1]
    cur_time = time.time()
    print(f"Done filtering TCR by content type: {cur_time - start_time} sec")

    print("Drop content type column and null rows")
    tcr = tcr.drop("content_type_abrg", axis=1)
    tcr = tcr.dropna(subset=["task_container_id", "part"])  # 243 rows
    tcr.rename(columns={"updated_at": "start_time"}, inplace=True)
    cur_time = time.time()
    print(f"Done dropping columns and rows: {cur_time - start_time} sec")

    print("Casting floats to ints: elapsed_time, task_container_id, part")
    tcr.elapsed_time_in_ms = tcr.elapsed_time_in_ms.astype(int)
    tcr.task_container_id = tcr.task_container_id.astype(int)
    tcr.part = tcr.part.astype(int)
    cur_time = time.time()
    print(f"Done casting to int: {cur_time - start_time} sec")

    print("Filter by in/out filters")
    print("In filters: uid in positions")
    print("Out filters: 1. admins 2. score_ids 3. japanese ids")
    tcr = tcr.query("student_id in @in_filters and student_id not in @out_filters")
    cur_time = time.time()
    print(f"Done filtering by in/out filters: {cur_time - start_time} sec")

    print("Sort by student_id and start_time")
    tcr = tcr.sort_values(["student_id", "start_time"])
    cur_time = time.time()
    print(f"Done sorting: {cur_time - start_time} sec")

    print("Add time_limits_in_ms column using time_limit.csv")
    print("and map NaN values to the average number on parts using time_limits_in_ms")
    time_limits = pd.read_csv("am_v2/tmp/time_limits.csv")
    part2qids = [
        tcr[tcr.part == part].content_id.unique().tolist() for part in range(1, 8)
    ]
    qid2part = {qid: part + 1 for part, qids in enumerate(part2qids) for qid in qids}
    avg_times = (
        time_limits.groupby("part_number").mean().astype(int).time_limit_in_ms.to_dict()
    )
    qid_diff = set(qid2part.keys()) - set(time_limits.question_id.unique())
    qid2avg = {qid: avg_times[qid2part[qid]] for qid in qid_diff}
    qid2limit = (
        time_limits.set_index("question_id").time_limit_in_ms.astype(int).to_dict()
    )
    qid2limit.update(qid2avg)
    tcr["time_limit_in_ms"] = tcr.content_id.map(qid2limit)

    print("Move 'part' column to index 5")
    col = tcr.pop("part")
    tcr.insert(5, col.name, col)
    print("Move 'time_limit_in_ms' column to index 7")
    col = tcr.pop("time_limit_in_ms")
    tcr.insert(7, col.name, col)

    print(f"Saving to '{BASE_PATH}/new_am_data.csv'")
    tcr.to_csv(f"{BASE_PATH}/new_am_data.csv", index=False)
    cur_time = time.time()
    print(f"Done saving: {cur_time - start_time} sec")
