import csv
import sys
import json
import codecs
import unicodedata
import numpy as np
import keras
import torch
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from sklearn.externals import joblib
from gensim import corpora, matutils
from keras.preprocessing.sequence import pad_sequences
import torch.utils.data as data_utils

import janome.tokenizer

janomet = janome.tokenizer.Tokenizer()


def format_word(word):
    """
    正規化
    """
    # 空白を削除
    result = word.strip()
    # NFKC（Normalization Form Compatibility Composition）で
    # 半角カタカナ、全角英数、ローマ数字・丸数字、異体字などなどを正規化。
    result = unicodedata.normalize("NFKC", result)
    result = result.lower()
    return result


def load_synonym_dict(dictname):
    """
    類義語辞書をロードする
    """
    synonyms = []
    with open(dictname, 'r', encoding="cp932") as f:
        i = 0
        for line in f:
            i += 1
            if i == 1:
                continue
            line = line.translate(str.maketrans({'\n': None, '"': None}))
            ws = line.split(',')
            cat = ws.pop(0)
            key = ws.pop(0)
            if len(cat) == 0:
                continue
            if len(key) > 0:
                ws.append(key)
            arr = []
            for w in ws:
                if len(w) > 0:
                    arr.append(w)
            synonyms.append([cat, key, arr])
    return synonyms


def match_syns(s, synonyms):
    """
    類義語とストップワードの処理
    """
    if len(s) > 0:
        for synonyms1 in synonyms:
            cat = synonyms1[0]
            synonyms1_key = synonyms1[1]
            synonyms1_val = synonyms1[2]
            if s in synonyms1_val:
                return synonyms1_key
    return s


def get_tokens(line, synonyms):
    """
    テキストをトークン化する
    """
    line = format_word(line)
    tokens = []
    for node in janomet.tokenize(line):
        part = node.part_of_speech.split(',')[0]
        t = ''
        if part in ['名詞']:
            t = match_syns(node.surface, synonyms)
        elif part in ['動詞', '副詞', '形容詞']:
            t = match_syns(node.base_form, synonyms)
        if len(t) > 0:
            tokens.append(t)
    return tokens


def get_dict(dict_path, words):
    """
    単語辞書をつくる
    """
    dictionary = corpora.Dictionary(words)
    dictionary.save_as_text(dict_path)
    return dictionary


def load_dict(dict_path):
    """
    単語辞書をロードする
    """
    dictionary = corpora.Dictionary.load_from_text(dict_path)
    return dictionary


def is_eos(w, v, i):
    """
    EOSを判定する
    """
    if len(v) > i + 2:
        # 項目 : 値 スキーマの処理
        if v[i+2] == ':':
            return True
    return w == '。' or w == '|' or w == '，'


def get_word(dictionary, line):
    """
    1行のトークンリストをIDリストにする
    """
    vec = dictionary.doc2idx(line, unknown_word_index=2)
    # word_to_id["<PAD>"] = 0
    # word_to_id["<START>"] = 1
    # word_to_id["<UNK>"] = 2
    # word_to_id['<EOS>'] = 3
    words = [1]
    i = 0
    for a in vec:
        words.append(a)
        # 文の終わりなら3をいれる
        w = line[i]
        # 文の終わりを検出
        if is_eos(w, line, i):
            words.append(3)
            words.append(1)
        i += 1
    words.append(3)
    return words


def load_data(data_params, max_len, vocab_size):
    data_path = data_params['data_csv']
    dict_path = data_params['dict_txt']
    syns_path = data_params['syns_csv']

    # 類義語辞書をロード
    synonyms = []
    if len(syns_path) > 0:
        synonyms = load_synonym_dict(syns_path)

    # データをロード
    full_dataset = []
    with open(data_path, 'r', encoding="cp932") as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            if len(line) > 0:
                full_dataset.append(line)
            i += 1

    # ランダムな20%のデータを検査データにする
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size])

    # それぞれのデータをラベル(数値)と入力データ(文字列)に分解
    x_train_data = []
    y_train_data = []
    x_test_data = []
    y_test_data = []

    for line in train_dataset:
        if len(line) == 2 and len(line[0]) > 0 and len(line[1]) > 0:
            y_train_data.append(int(line[0]))
            x_train_data.append(line[1])
    for line in test_dataset:
        if len(line) == 2 and len(line[0]) > 0 and len(line[1]) > 0:
            y_test_data.append(int(line[0]))
            x_test_data.append(line[1])

    # 入力データ(文字列)を形態素解析
    # 存在する全単語をwords配列に入れる
    words = [["<PAD>", "<START>", "<UNK>", '<EOS>']]
    train_words = []
    for line in x_train_data:
        a = get_tokens(line, synonyms)
        words.append(a)
        train_words.append(a)
    test_words = []
    for line in x_test_data:
        a = get_tokens(line, synonyms)
        words.append(a)
        test_words.append(a)

    # 存在する全単語から辞書を作成
    dictionary = get_dict(dict_path, words)
    num_words = len(dictionary)

    # 辞書を使って形態素を数値化
    num_wpl = 0
    x_train = []
    for line in train_words:
        a = get_word(dictionary, line)
        x_train.append(a)
        if len(a) > num_wpl:
            num_wpl = len(a)
    x_test = []
    for line in test_words:
        a = get_word(dictionary, line)
        x_test.append(a)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train_data = np.array(y_train_data)
    y_test_data = np.array(y_test_data)

    print(num_words, 'words ->', vocab_size)
    print(num_wpl, 'words/line (max) ->', max_len)

    return (x_train, y_train_data), (x_test, y_test_data), dictionary.token2id


def load_label_data(dictname):
    """
    ラベルをロードする
    """
    labels = []
    with open(dictname, 'r', encoding="cp932") as f:
        for line in f:
            line = line.translate(str.maketrans({'\n': None, '"': None}))
            ws = line.split(',')
            labels.append(ws[0])
    return labels


def load_reuters_data(vocab_size):
    INDEX_FROM = 3
    train_set, test_set = reuters.load_data(
        path="reuters.npz", num_words=vocab_size, skip_top=0, index_from=INDEX_FROM)
    word_to_id = reuters.get_word_index(path="reuters_word_index.json")
    word_to_id = {k: (v+3) for k, v in word_to_id.items()}
    id_to_word = {value: key for key, value in word_to_id.items()}
    return train_set, test_set, word_to_id


def load_data_from_file(vocab_size):
    new_array = np.load('data.npz')
    data = new_array['x']
    label = new_array['y']

    f = open('data.json', 'r', encoding="utf8")
    word_to_id = json.load(f)
    print(word_to_id)

    return (data, label), (data, label), word_to_id


def save_data_to_file(data_params):
    (x_train, y_train), (x_test, y_test), word_to_id = load_data(
        data_params, 2000, 20000)
    np.savez('data.npz', x=x_train, y=y_train)
    f = codecs.open('data.json', 'w', 'utf-8')
    json.dump(word_to_id, f, ensure_ascii=False)


def load_data_set(data_params, type, max_len, vocab_size, batch_size):

    print('\nLoading data...')
    train_set, test_set, word_to_id = load_data(
        data_params, max_len, vocab_size)

    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    word_to_id['<EOS>'] = 3

    x_train, y_train = train_set[0], train_set[1]
    x_test, y_test = test_set[0], test_set[1]
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    # debug: 分割結果を1つ表示する
    # id_to_word = {value: key for key, value in word_to_id.items()}
    # print(str(y_test[0]) + " ".join([id_to_word.get(i) for i in x_test[0]]))

    x_train_pad = pad_sequences(
        x_train, maxlen=max_len, padding='post', truncating='post', value=0.0)
    x_test_pad = pad_sequences(
        x_test, maxlen=max_len, padding='post', truncating='post', value=0.0)

    # debug: 分割結果を1つ表示する
    # for test in x_test_pad:
    #     print(" ".join([id_to_word.get(i) for i in test]))
    #     break

    train_data = data_utils.TensorDataset(torch.from_numpy(x_train_pad).type(
        torch.LongTensor), torch.from_numpy(y_train).type(torch.LongTensor))

    train_loader = data_utils.DataLoader(
        train_data, batch_size=batch_size, drop_last=True)

    return train_loader, train_set, test_set, x_train_pad, x_test_pad, word_to_id
