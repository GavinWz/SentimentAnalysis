import json
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

token_dic = json.load(open('datas/tokens.json', 'r', encoding='utf-8'))
df = pickle.load(open('datas/eval.pkl', 'rb'))
print(df)
data = df[-1]
for id in data[0]:
    print(token_dic[id], end=' ')

# df = pd.read_csv('datas/splitted_data.csv')
# train_datas, test_datas = train_test_split(df.to_numpy(), test_size=0.2, random_state=42)
# print(train_datas[-1])
