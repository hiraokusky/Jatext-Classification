# Jatext Classificator

日本語テキストの分類を実行するツールです。

## 対応している分類器

### Structured Self-attentive sentence embeddings

Bi-LSTMと注意機構でテキストを分類します。
二項分類と多項分類に対応しています。(Jatext-Classificatorではまだ多項分類にしか対応していません）

* More detail

  https://github.com/kaushalshetty/Structured-Self-Attention

## 使い方

### Train

入力データファイルのうち、ランダムに80%を選び、学習データとして用います。
残りの20%を検証データとして用います。

```
python train.py --data_csv 入力データファイルのパス [--syns_csv 類義語辞書ファイルのパス] [--dict_txt キャッシュ辞書ファイルのパス] [--num_visuals ビジュアル化するテスト数]
```

### Predict

T.B.D.

### Visualize

fork元の実装通り、注意された単語を可視化したHTMLファイルattention.htmlを生成します。
検証データの中から、num_visualsで指定した件数分を表示します。

## 入力データファイル

分類とテキストを記載したファイルです。
データに応じて、以下のパラメータを調整してください。

batch_size, vocab_size, timesteps, num_classes

フォーマット: ヘッダー無し
```
分類番号, テキスト
```

例:
```
1, 今日はいい天気ですね。
2, 調子が悪くなった。
```

## 類義語辞書ファイル

入力データを形態素解析した結果に対して、類義語と一致する単語を見出し語に変換する辞書です。
見出し語を空文字にすることで、ストップワードを実現できます。

フォーマット: ヘッダー有り
```
分類, 見出し語, 類義度1, ...
```

例:
```
分類, 見出し語, 類義語1, 類義語2
ストップワード,,こと,よう
経過,悪化,悪くなった
```

## キャッシュ辞書ファイル

現在は使っていません。

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

