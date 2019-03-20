import codecs
from gensim.models import word2vec
import numpy as np
import torch
import unicodedata
import neologdn
import re
import csv
from gensim import corpora, matutils

class JDistribution:
    """
    トークン列をコーパスを用いて分散表現に変換する
    """

    embedding_dim = 200

    def save_model(self, path, corpus):
        """
        word2vecモデルをつくる
        """
        # corpus = [sentence.split() for sentence in corpus]
        # size: 単語ベクトルの次元数	
        # min_count: n回未満登場する単語を破棄
        # window: 学習に使う前後の単語数
        self.model = word2vec.Word2Vec(corpus, size=self.embedding_dim, min_count=0, window=3)
        self.model.save(path)

    def load_model(self, path):
        self.model = word2vec.Word2Vec.load(path)
        return self.model

    def embeddings(self, word_to_id):
        """
        embeddings[id] = ベクトル
        """
        n = 0
        embeddings = np.zeros((len(word_to_id), self.embedding_dim))
        for k, v in word_to_id.items():
            if k in self.model.wv:
                n += 1
                vector = np.array(self.model.wv[k], dtype='float32')
            else:
                print(k)
                vector = np.zeros(self.embedding_dim)
            embeddings[v] = vector
        print(n, '/', len(word_to_id.items()))
        return torch.from_numpy(embeddings).float()

    def save_dict(self, dict_path, words):
        """
        単語辞書をつくる
        """
        dictionary = corpora.Dictionary(words)
        dictionary.save_as_text(dict_path)
        return dictionary

    def load_dict(self, dict_path):
        """
        単語辞書をロードする
        """
        dictionary = corpora.Dictionary.load_from_text(dict_path)
        return dictionary

    def create_model(self, data_path):
        """
        形態素解析済みコーパスをロードしてword2vecモデルをつくる
        """
        # テキストファイルを行ごとにしたリストを形態素解析してトークンリストにする
        lines = [["<PAD>", "<START>", "<UNK>", '<EOS>']]
        with open(data_path, 'r', encoding="cp932") as f:
            reader = csv.reader(f)
            i = 0
            for line in reader:
                lines.append(line[1].split())

        # コーパスにword2vecをかけてモデルとして保存する
        self.save_model('db/corpus.model', lines)

    def create_dictionary(self, dict_path, data_path):
        """
        word2vecモデルをロードして利用している単語についてのベクトルマップを作る
        """
        lines = [["<PAD>", "<START>", "<UNK>", '<EOS>']]
        with open(data_path, 'r', encoding="cp932") as f:
            reader = csv.reader(f)
            i = 0
            for line in reader:
                lines.append(line[1].split())
        dictionary = self.save_dict(dict_path, lines)
        return dictionary
        
    def load_embeddings(self, dict_path):
        dictionary = self.load_dict(dict_path)
        self.load_model('db/corpus.model')

        # word2vecのベクトルを辞書ファイルのidと結びつける
        embeddings = self.embeddings(dictionary.token2id)

        # これを使ってLSTMする
        return embeddings

    def get_synonyms(self, text):
        model = self.load_model('db/corpus.model')
        results = model.wv.most_similar(positive=[text])
        return results

# import datetime
# jd = JDistribution()
# now = datetime.datetime.now()
# print(now)
# jd.create('db/corpus0-10000.csv')
# end = datetime.datetime.now()
# print(end)
# print(end - now)

# embeddings = jd.load_embeddings('db/words.txt')
# for e in embeddings:
#     print(e)
# load()

# words = []
# for k in dictionary.token2id:
#     words.append(k)
# print(words)

# for k in words:
#     print(k, corpus.model.wv[k])
