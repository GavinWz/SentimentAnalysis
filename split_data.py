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
    split_data = []  # store the splited data
    token_count_dic = {}  # store all tokens and their count
    for review, sentiment in tqdm(df.values):
        review = review.replace(r'<br /><br />',
                                '').strip()  # delete '<br /><br />' and blank characters at head and tail
        review_tokens = word_tokenize(review)  # cut sentence into tokens
        label = 1 if sentiment == 'positive' else 0  # create label number
        for token in review_tokens:  # calculate frequency of every token
            token_count_dic[token] = token_count_dic.get(token, 0) + 1
        split_data.append((' '.join(review_tokens), label))  # fill split data list

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_count_list = list(token_count_dic.items())
    token_count_list.sort(key=lambda x: x[1], reverse=True)  # sort tokens by their frequency
    tokens = [token for token, count in token_count_list if
              count >= 5]  # ignore tokens with a frequency smaller than 5
    for token in ['<NUM>', '<PUN>', '<UNK>', '<PAD>']:  # add special tokens
        tokens.insert(0, token)
    with open(str(output_dir / 'tokens.json'), 'w', encoding='utf-8') as writer:  # save token set as a json file
        json.dump(tokens, writer, ensure_ascii=False, sort_keys=True, indent=2)
    df = pd.DataFrame(split_data,
                      columns=['review', 'label'])  # convert split data to pandas Dataframe and save as csv file
    df.to_csv((str(output_dir / 'split_data.csv')), index=False)


def is_number(text):  # judge text is number or not
    re_num = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
    return bool(re_num.match(text))


def is_punctuation(text):# judge text is punctuation or not
    pattern = r'^\W$'
    if re.match(pattern, text):
        return True
    else:
        return False


def split_dataset(dataset_dir, require_val=True):
    '''
    Split dataset and save as binary files
    :param dataset_dir: dataset directory
    :param require_val: whether require validation dataset
    :return:
    '''
    tokens = json.load(open(dataset_dir + '/tokens.json', 'r', encoding='utf-8'))
    token_id_dic = dict(zip(tokens, range(len(tokens))))  # create token id dictionary
    unk_id = token_id_dic['<UNK>']

    dataset_file = dataset_dir + '/split_data.csv'
    df = pd.read_csv(dataset_file)  # read split data, df:[review, label]
    datas = []
    for review, label in tqdm(df.values, desc=''):
        tokens = review.split(' ')  # split review into tokens
        token_ids = []

        # convert tokens to ids
        for token in tokens:
            if is_number(token):
                token_ids.append(token_id_dic['<NUM>'])
            elif is_punctuation(token):
                token_ids.append(token_id_dic['<PUN>'])
            else:
                token_ids.append(token_id_dic.get(token, unk_id))

        # consider token ids as a feature, combine it with a label, and add to final dataset
        datas.append([token_ids, label])

    # split final dataset into training data and testing data
    train_datas, test_datas = train_test_split(datas, test_size=0.2, random_state=42)

    # if you want a validation set, then split training set into new training set and validation set
    if require_val:
        train_datas, eval_datas = train_test_split(train_datas, test_size=0.2, random_state=42)

    print(f'Size of training set: {len(train_datas)}')
    print(f'Size of testing set: {len(eval_datas)}')
    if require_val:
        print(f'Size of validation set: {len(test_datas)}')

    # save split datasets as binary files
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
