# You can write your own classification file to use the module
from attention.model import StructuredSelfAttention
from attention.train import train, get_activation_wts, evaluate, predict
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
import csv

from input.data_loader import load_data_set, load_label_data

import argparse

class Predict:

    data_params = {}
    model_params = {}
    params_set = {}

    def json_to_dict(self, json_set):
        for k, v in json_set.items():
            if v == 'False':
                json_set[k] = False
            elif v == 'True':
                json_set[k] = True
            else:
                json_set[k] = v
        return json_set

    def init_model(self):
        with open('config.json', 'r') as f:
            self.params_set = json.load(f)

        with open('model_params.json', 'r') as f:
            self.model_params = json.load(f)

        self.params_set = self.json_to_dict(self.params_set)
        self.model_params = self.json_to_dict(self.model_params)

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
            self.data_params['data_csv'] = args.data_csv
        else:
            self.data_params['data_csv'] = 'data.csv'
        if args.labels_csv != None and len(args.labels_csv) > 0:
            self.data_params['labels_csv'] = args.labels_csv
        else:
            self.data_params['labels_csv'] = 'labels.csv'
        if args.syns_csv != None and len(args.syns_csv) > 0:
            self.data_params['syns_csv'] = args.syns_csv
        else:
            self.data_params['syns_csv'] = ''

        if args.dict_txt != None and len(args.dict_txt) > 0:
            self.data_params['dict_txt'] = args.dict_txt
        else:
            self.data_params['dict_txt'] = 'dict.txt'

        self.params_set['verbose'] = args.verbose

        print('\nLoading settings...')
        print("data :", self.data_params)
        print("param:", self.params_set)
        print("model:", self.model_params)

    def predict_attention(self, attention_model, wts, x_test_pad, word_to_id, word_to_word, count, filename):
        labels = load_label_data(self.data_params['labels_csv'])

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
                if m >= count:
                    break

            result.append(pred)
            n += 1

        return result

    def do_predict(self, text, count):
        PATH = 'db/jc.model'
        MAXLENGTH = self.model_params['timesteps']

        # Load data
        full_dataset = [['0', text]]
        # data_path = data_params['data_csv']
        # full_dataset = []
        # with open(data_path, 'r', encoding="cp932") as f:
        #     reader = csv.reader(f)
        #     i = 0
        #     for line in reader:
        #         if len(line) > 0:
        #             full_dataset.append(line)
        #         i += 1
        #         break
        # print(full_dataset)

        train_loader, train_set, test_set, x_train_pad, x_test_pad, word_to_id, word_to_word = load_data_set(
            full_dataset, self.data_params, 1, MAXLENGTH, self.model_params["vocab_size"], self.model_params['batch_size'], True)

        # Using pretrained embeddings
        if self.params_set["use_embeddings"]:
            embeddings = load_glove_embeddings(
                "glove/glove.6B.50d.txt", word_to_id, 50)
        else:
            embeddings = None

        # Loadl model
        attention_model = StructuredSelfAttention(batch_size=train_loader.batch_size, lstm_hid_dim=self.model_params['lstm_hidden_dimension'], d_a=self.model_params["d_a"], r=self.params_set["attention_hops"], vocab_size=len(
            word_to_id), max_len=MAXLENGTH, type=1, n_classes=self.model_params["num_classes"], use_pretrained_embeddings=self.params_set["use_embeddings"], embeddings=embeddings)
        attention_model.load_state_dict(torch.load(PATH))

        # Predict
        wts = get_activation_wts(attention_model, Variable(
            torch.from_numpy(x_test_pad[:]).type(torch.LongTensor)))
        print(wts.size())
        res = self.predict_attention(
            attention_model, wts, x_test_pad[:], word_to_id, word_to_word, count, filename='predict_attention')
        return res

if __name__ == '__main__':
    predict = Predict()
    predict.init_model()
    
    while True:
        inputstr = input('> ')
        if len(inputstr) > 0:
            res = predict.do_predict(inputstr, 10)
            print(res)
