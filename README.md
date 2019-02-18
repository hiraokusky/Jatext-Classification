# Jatext Classificator

日本語テキストの分類を実行するツールです。

## 対応している分類器

### Structured Self-attentive sentence embeddings

Bi-LSTMと注意機構でテキストを分類します。
二項分類と多項分類に対応しています。

* More detail

  https://github.com/kaushalshetty/Structured-Self-Attention

## 使い方

### Train

```
python train.py --data_csv 学習データファイルのパス [--dict_txt キャッシュ辞書データファイルのパス]
```

### Predict

T.B.D.

### Visualize

attention.htmlをブラウザで開きます。

## 学習データ

フォーマット: ヘッダー無し
```
分類番号, テキスト
```

例:
```
1, 今日はいい天気ですね。
2, 調子が悪くなった。
```

## config.json

	"epochs":2,
	"use_regularization":"True",
	"C":0.03,
	"clip":"True",
	"use_embeddings":"False",
	"attention_hops":10

* epochs

  学習の回数です。

## model_params.json

	"batch_size":16,
	"vocab_size":20000,
	"timesteps":200,
	"lstm_hidden_dimension":50,
	"d_a":100,
	"num_classes": 112

* batch_size

  バッチサイズです。
  データ数より小さくなくてはいけません。

* vocab_size

  扱う単語の種類数です。

* timesteps

  1テキストで扱う単語の数です。

* num_classes

  分類するクラス数です。
  data_csvファイルの分類数と同じでないといけません。

* lstm_hidden_dimension

  LSTMの隠れ層の次元数です。

