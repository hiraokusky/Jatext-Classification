# You can write your own classification file to use the module
from attention.model import StructuredSelfAttention
from attention.train import train, get_activation_wts, evaluate
from utils.pretrained_glove_embeddings import load_glove_embeddings
from visualization.attention_visualization import createHTML
import torch
import numpy as np
from torch.autograd import Variable
from keras.preprocessing.sequence import pad_sequences
import torch.nn.functional as F
import torch.utils.data as data_utils
import os
import sys
import json

from input.data_loader import load_data_set, load_label_data

import argparse

classified = False


def json_to_dict(json_set):
    for k, v in json_set.items():
        if v == 'False':
            json_set[k] = False
        elif v == 'True':
            json_set[k] = True
        else:
            json_set[k] = v
    return json_set


with open('config.json', 'r') as f:
    params_set = json.load(f)

with open('model_params.json', 'r') as f:
    model_params = json.load(f)

params_set = json_to_dict(params_set)
model_params = json_to_dict(model_params)

data_params = {}

parser = argparse.ArgumentParser(
    prog='train'
)
# parser.add_argument("square", type=int,
#                     help="display a square of a given number")
parser.add_argument("-i", "--data_csv", type=str,
                    help="data_csv")
parser.add_argument("-d", "--dict_txt", type=str,
                    help="dict_txt")
parser.add_argument("-s", "--syns_csv", type=str,
                    help="syns_csv")
parser.add_argument("-l", "--labels_csv", type=str,
                    help="labels_csv")

parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()

if args.data_csv != None and len(args.data_csv) > 0:
    data_params['data_csv'] = args.data_csv
else:
    data_params['data_csv'] = 'data.csv'
if args.labels_csv != None and len(args.labels_csv) > 0:
    data_params['labels_csv'] = args.labels_csv
else:
    data_params['labels_csv'] = 'labels.csv'
if args.syns_csv != None and len(args.syns_csv) > 0:
    data_params['syns_csv'] = args.syns_csv
else:
    data_params['syns_csv'] = ''

if args.dict_txt != None and len(args.dict_txt) > 0:
    data_params['dict_txt'] = args.dict_txt
else:
    data_params['dict_txt'] = 'dict.txt'

params_set['verbose'] = args.verbose

print('\nLoading settings...')
print("data :", data_params)
print("param:", params_set)
print("model:", model_params)

# classification_type = sys.argv[1]
classification_type = 'multiclass'
if model_params["num_classes"] <= 2:
    classification_type = 'binary'


def visualize_attention(attention_model, wts, x_test_pad, word_to_id, y_test, filename):
    print(filename, "{} samples".format(len(x_test_pad)))

    labels = load_label_data(data_params['labels_csv'])

    wts_add = torch.sum(wts, 1)
    wts_add_np = wts_add.data.numpy()
    wts_add_list = wts_add_np.tolist()
    id_to_word = {v: k for k, v in word_to_id.items()}
    result = []
    text = []
    correct = 0
    correct2 = 0
    n = 0
    for test in x_test_pad:
        attention_model.batch_size = 1
        attention_model.hidden_state = attention_model.init_hidden()
        x_test_var = Variable(torch.from_numpy(test).type(torch.LongTensor))
        y_test_pred, _ = attention_model(x_test_var)

        # 結果のリストを降順に並べる
        m = 0
        dic = {}
        for x in y_test_pred[0]:
            dic[x] = m
            m += 1
        yy = sorted(dic.items(), reverse=True)
        m = 0
        pred = []
        for y in yy:
            m += 1
            l = str(y[1])
            if len(labels) > 0:
                l = labels[y[1]]
            pred.append(l)
            if m > 5:
                break

        l = str(y_test[n])
        if len(labels) > 0:
            l = labels[y_test[n]]

        if l == pred[0]:
            correct += 1
        for r in pred:
            if l == r:
                correct2 += 1
                break

        text.append(" ".join([id_to_word.get(i) for i in test]))
        result.append([l, pred[0], pred])
        n += 1

    # print(text[0])

    # 20個ずつhtmlに出力
    m = 20
    n = len(result)
    for i in range(0, n, m):
        j = i + m
        createHTML(result[i:j], text[i:j], wts_add_list[i:j],
                   filename + '_' + str(i + 1) + '_' + str(j) + '.html')

    return (correct, correct2, result)


def binary_classfication(attention_model, train_loader, epochs=5, use_regularization=True, C=1.0, clip=True):
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.RMSprop(attention_model.parameters())
    train(params_set, attention_model, train_loader, loss,
          optimizer, epochs, use_regularization, C, clip)


def multiclass_classification(attention_model, train_loader, epochs=5, use_regularization=True, C=1.0, clip=True):
    loss = torch.nn.NLLLoss()
    optimizer = torch.optim.RMSprop(attention_model.parameters())
    train(params_set, attention_model, train_loader, loss,
          optimizer, epochs, use_regularization, C, clip)


MAXLENGTH = model_params['timesteps']
if classification_type == 'binary':

    train_loader, x_test_pad, y_test, word_to_id = load_data_set(
        data_params, 0, MAXLENGTH, model_params["vocab_size"], model_params['batch_size'])  # loading imdb dataset

    if params_set["use_embeddings"]:
        embeddings = load_glove_embeddings(
            "glove/glove.6B.50d.txt", word_to_id, 50)
    else:
        embeddings = None
    # Can use pretrained embeddings by passing in the embeddings and setting the use_pretrained_embeddings=True
    attention_model = StructuredSelfAttention(batch_size=train_loader.batch_size, lstm_hid_dim=model_params['lstm_hidden_dimension'], d_a=model_params["d_a"], r=params_set["attention_hops"], vocab_size=len(
        word_to_id), max_len=MAXLENGTH, type=0, n_classes=1, use_pretrained_embeddings=params_set["use_embeddings"], embeddings=embeddings)

    # Can set use_regularization=True for penalization and clip=True for gradient clipping
    binary_classfication(attention_model, train_loader=train_loader,
                         epochs=params_set["epochs"], use_regularization=params_set["use_regularization"], C=params_set["C"], clip=params_set["clip"])
    classified = True
    #wts = get_activation_wts(binary_attention_model,Variable(torch.from_numpy(x_test_pad[:]).type(torch.LongTensor)))
    #print("Attention weights for the testing data in binary classification are:",wts)


if classification_type == 'multiclass':

    train_loader, train_set, test_set, x_train_pad, x_test_pad, word_to_id = load_data_set(
        data_params, 1, MAXLENGTH, model_params["vocab_size"], model_params['batch_size'])

    # Using pretrained embeddings
    if params_set["use_embeddings"]:
        embeddings = load_glove_embeddings(
            "glove/glove.6B.50d.txt", word_to_id, 50)
    else:
        embeddings = None
    attention_model = StructuredSelfAttention(batch_size=train_loader.batch_size, lstm_hid_dim=model_params['lstm_hidden_dimension'], d_a=model_params["d_a"], r=params_set["attention_hops"], vocab_size=len(
        word_to_id), max_len=MAXLENGTH, type=1, n_classes=model_params["num_classes"], use_pretrained_embeddings=params_set["use_embeddings"], embeddings=embeddings)

    # Using regularization and gradient clipping at 0.5 (currently unparameterized)
    multiclass_classification(attention_model, train_loader,
                              epochs=params_set["epochs"], use_regularization=params_set["use_regularization"], C=params_set["C"], clip=params_set["clip"])
    classified = True

    #wts = get_activation_wts(multiclass_attention_model,Variable(torch.from_numpy(x_test_pad[:]).type(torch.LongTensor)))
    #print("Attention weights for the data in multiclass classification are:",wts)

if classified:
    print('\nVisualizing...')

    wts = get_activation_wts(attention_model, Variable(
        torch.from_numpy(x_train_pad[:]).type(torch.LongTensor)))
    print(wts.size())
    (train_corrent, train_corrent2, train_result) = visualize_attention(
        attention_model, wts, x_train_pad[:], word_to_id, train_set[1], filename='train_attention')

    wts = get_activation_wts(attention_model, Variable(
        torch.from_numpy(x_test_pad[:]).type(torch.LongTensor)))
    print(wts.size())
    (test_corrent, test_corrent2, test_result) = visualize_attention(
        attention_model, wts, x_test_pad[:], word_to_id, test_set[1], filename='test_attention')

    print('Result: train')
    for r in train_result:
        print(r[0] == r[1], r[0] in r[2], r[0], r[2]
              [0], r[2][1], r[2][2], r[2][3], r[2][4])

    print('Result: test')
    for r in test_result:
        print(r[0] == r[1], r[0] in r[2], r[0], r[2]
              [0], r[2][1], r[2][2], r[2][3], r[2][4])

    print('Correct:')
    print('test  acc:', test_corrent / len(x_test_pad[:]))
    print('train acc:', train_corrent / len(x_train_pad[:]))
    print('total acc:', (train_corrent + test_corrent) /
          (len(x_train_pad[:]) + len(x_test_pad[:])))

    print('In 5th:')
    print('test  acc:', test_corrent2 / len(x_test_pad[:]))
    print('train acc:', train_corrent2 / len(x_train_pad[:]))
    print('total acc:', (train_corrent2 + test_corrent2) /
          (len(x_train_pad[:]) + len(x_test_pad[:])))
