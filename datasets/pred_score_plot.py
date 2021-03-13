import pickle
import numpy as np
import matplotlib.pyplot as plt


with open("/private/results/results.pkl", "rb") as f_r:
    results = pickle.load(f_r)


data = {i: [] for i in range(0, 1000, 10)}
for cross_num, result in results.items():
    for i, l1 in enumerate(result["l1"]):
        if result["label"][i] == 4945:
            continue
        data[int(result["label"][i] / 10) * 10].append(l1)

for label, l1_list in data.items():
    if len(l1_list) == 0:
        data[label] = 0
    else:
        data[label] = np.mean(l1_list)

data_list = list(data.items())
data_list.sort()
x, y = zip(*data_list)
plt.figure()
plt.bar(x, y, width=10, color="firebrick")
plt.xlabel("Score", fontsize=15)
plt.ylabel("MAE", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
