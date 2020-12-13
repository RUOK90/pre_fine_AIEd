"""Script to split data
"""
import dill as pkl
import pandas as pd

from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    pk2id = (
        pd.read_csv("/shared/new_am_data/students_final_ver2.csv")
        .set_index("pk")
        .id.to_dict()
    )

    AM_DATA_PATH = "/shared/modified_AAAI20/response"
    # score_data_path = "/shared/AAAI20/score_data/14d_10q"

    am_train = pd.read_csv(
        f"{AM_DATA_PATH}/train_user_list.csv", header=None, names=["pk"]
    )
    am_val = pd.read_csv(f"{AM_DATA_PATH}/val_user_list.csv", header=None, names=["pk"])
    am_test = pd.read_csv(
        f"{AM_DATA_PATH}/test_user_list.csv", header=None, names=["pk"]
    )
    am_train.pk = am_train.pk.apply(lambda x: x.split(".")[0])
    am_val.pk = am_val.pk.apply(lambda x: x.split(".")[0])
    am_test.pk = am_test.pk.apply(lambda x: x.split(".")[0])
    am_train.pk = am_train.pk.map(pk2id)
    am_val.pk = am_val.pk.map(pk2id)
    am_test.pk = am_test.pk.map(pk2id)
    print(len(am_train) + len(am_val) + len(am_test))
    print(len(am_train), len(am_val), len(am_test))
    print(
        am_train.isnull().values.any(),
        am_val.isnull().values.any(),
        am_test.isnull().values.any(),
    )
    am_users = (
        set(am_train.pk.unique()) | set(am_val.pk.unique()) | set(am_test.pk.unique())
    )
    old_am_users = [
        list(am_train.pk.unique()),
        list(am_val.pk.unique()),
        list(am_test.pk.unique()),
    ]
    am_train_users, am_test_users = train_test_split(list(am_users), test_size=0.05)
    am_val_users, am_test_users = train_test_split(list(am_test_users), test_size=0.5)
    print(len(am_train_users), len(am_val_users), len(am_test_users))

    with open("am_v2/load/user_split.pkl", "wb") as output_file:
        pkl.dump([am_train_users, am_val_users, am_test_users], output_file)
    with open("am_v2/load/old_user_split.pkl", "wb") as output_file:
        pkl.dump(old_am_users, output_file)
