#%%
import pandas as pd

df = pd.read_csv(r"C:\Users\default.DESKTOP-5TT5SG8\Desktop\cw\통합 문서1.csv")
print(df.head())
#%%
# print(df.keys())
out_1 = []
idx = []

for i, (t, v1) in enumerate(zip(df["A Man's Turf: The Perfect Lawn"], df["1"])):
    if int(v1) >= 10:
        out_1.append(t)
        idx.append(i)
        
print(len(out_1))
s_out_1 = set(out_1)
print(len(s_out_1))
#%%
print(out_1)
#%%
data = pd.read_csv(r"C:\Users\default.DESKTOP-5TT5SG8\Downloads\jbnu-swuniv-ai\train_data.csv")
#%%
row_idx = []
for i, title in enumerate(data["Title"]):
    for out in s_out_1:
        if title == out:
            row_idx.append(i)
print(len(row_idx))
#%%
data_filter = data.drop(row_idx, axis=0)
#%%
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(data_filter, test_size=0.1, shuffle=True)
#%%
# 기존 68682, filter 201, train: 67453, val: 1028 ver1
# 기존 68682, filter 96, train: 67557, val: 1029 ver2

print(len(train_data))
print(len(val_data))

print(len(train_data) + len(val_data))
#%%
train_data.to_csv("./Real_train_book_filter2_data_ver2.csv", index=False)
val_data.to_csv("./Real_val_book_filter2_data_ver2.csv", index=False)
#%%
# labels = {}

# for label in data_filter['label']:
#     if label not in labels:
#         labels[label] = 1
#     else:
#         labels[label] += 1
# #%%
# print(labels)