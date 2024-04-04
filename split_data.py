import json
import pickle
import re
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk.tokenize import word_tokenize


def preprocessing(dataset_file, output_dir):
    df = pd.read_csv(dataset_file)
    splitted_data = []  # store the splited data
    token_count_dic = {}  # store all tokens and their count
    for review, sentiment in tqdm(df.values):
        review = review.replace(r'<br /><br />',
                                '').strip()  # delete '<br /><br />' and blank characters at head and tail
        review_tokens = word_tokenize(review)  # cut sentence into tokens
        label = 1 if sentiment == 'positive' else 0  # create label number
        for token in review_tokens:  # calculate frequency of every token
            token_count_dic[token] = token_count_dic.get(token, 0) + 1
        splitted_data.append((' '.join(review_tokens), label))  # fill splitted data list

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_count_list = list(token_count_dic.items())
    token_count_list.sort(key=lambda x: x[1], reverse=True)
    tokens = [token for token, count in token_count_list if count >= 5]
    for token in ['<NUM>', '<PUN>', '<UNK>', '<PAD>']:
        tokens.insert(0, token)
    with open(str(output_dir / 'tokens.json'), 'w', encoding='utf-8') as writer:
        json.dump(tokens, writer, ensure_ascii=False, sort_keys=True, indent=2)
    df = pd.DataFrame(splitted_data, columns=['review', 'label'])
    df.to_csv((str(output_dir / 'splitted_data.csv')), index=False)

def is_number(num):
    re_num = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
    return bool(re_num.match(num))


def is_punctuation(char):
    pattern = r'^\W$'
    if re.match(pattern, char):
        return True
    else:
        return False

def split_dataset(dataset_dir):
    tokens = json.load(open(dataset_dir + '/tokens.json', 'r', encoding='utf-8'))
    token_id_dic = dict(zip(tokens, range(len(tokens))))  # create token and id dict
    unk_id = token_id_dic['<UNK>']

    dataset_file = dataset_dir + '/splitted_data.csv'
    df = pd.read_csv(dataset_file)
    datas = []
    for review, label in tqdm(df.values):
        tokens = review.split(' ')
        token_ids =  []
        for token in tokens:
            if is_number(token):
                token_ids.append(token_id_dic['<NUM>'])
            elif is_punctuation(token):
                token_ids.append(token_id_dic['<PUN>'])
            else:
                token_ids.append(token_id_dic.get(token, unk_id))
        datas.append([token_ids, label])

    train_datas, test_datas = train_test_split(datas, test_size=0.2, random_state=42)
    train_datas, eval_datas = train_test_split(train_datas, test_size=0.2, random_state=42)
    print(f'训练数据量：{len(train_datas)}')
    print(f'验证数据量：{len(eval_datas)}')
    print(f'测试数据量：{len(test_datas)}')

    with open(dataset_dir + '/train.pkl', 'wb') as writer:
        pickle.dump(train_datas, writer)
    with open(dataset_dir + '/eval.pkl', 'wb') as writer:
        pickle.dump(eval_datas, writer)
    with open(dataset_dir + '/test.pkl', 'wb') as writer:
        pickle.dump(test_datas, writer)

if __name__ == '__main__':
    dataset_file = 'IMDB_Dataset.csv'
    output_dir = './datas'
    preprocessing(dataset_file, output_dir)
    split_dataset(output_dir)