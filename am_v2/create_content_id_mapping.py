"""Script that creates question id mapping
to sequential indices as there are missing values
"""
import numpy as np
import pandas as pd


if __name__ == "__main__":
    data = pd.read_csv("/shared/new_am_data/new_am_data.csv")

    qids = sorted(data.content_id.unique())
    idx = np.arange(len(qids))

    new_mapping = {qid: i for qid, i in zip(qids, idx)}
    pd.DataFrame(pd.Series(new_mapping)).to_csv(
        "am_v2/load/new_content_dict.csv", index=True, header=False
    )
