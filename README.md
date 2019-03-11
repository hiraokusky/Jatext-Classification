# Jatext Classificator

日本語テキストの分類を実行するツールです。

## 対応している分類器

### Structured Self-attentive sentence embeddings

https://github.com/kaushalshetty/Structured-Self-Attention

* Bi-LSTMと注意機構でテキストを分類します。
* 二項分類と多項分類に対応しています。(Jatext-Classificatorではまだ多項分類にしか対応していません）

## 使い方

### Install

* NEologd対応版janomeが必要です。
* 以下の手順でインストールできます。

[neologd 辞書内包の janome パッケージをダウンロードできるようにしました（不定期更新）](https://medium.com/@mocobeta/neologd-%E8%BE%9E%E6%9B%B8%E5%86%85%E5%8C%85%E3%81%AE-janome-%E3%83%91%E3%83%83%E3%82%B1%E3%83%BC%E3%82%B8%E3%81%AE%E3%83%80%E3%82%A6%E3%83%B3%E3%83%AD%E3%83%BC%E3%83%89%E3%81%A7%E3%81%8D%E3%82%8B%E3%82%88%E3%81%86%E3%81%AB%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F-%E4%B8%8D%E5%AE%9A%E6%9C%9F%E6%9B%B4%E6%96%B0-71611ab66415)

### Train

* 入力データファイルのうち、ランダムに80%を選び、学習データとして用います。
* 残りの20%を検証データとして用います。

```
python train.py --data_csv 入力データファイルのパス [--syns_csv 類義語辞書ファイルのパス] [--dict_txt キャッシュ辞書ファイルのパス]  [--labels_csv ラベルデータファイル]
```

### Predict

* コマンドラインで入力したテキストから予測される分類結果を表示します。
* 最大10個の予測結果を確度の高い順に表示します。

```
python predict.py --data_csv 入力データファイルのパス [--syns_csv 類義語辞書ファイルのパス] [--dict_txt キャッシュ辞書ファイルのパス]  [--labels_csv ラベルデータファイル]
> 入力テキスト
[ 予測結果1, ... ]
```

### Visualize

* fork元の実装通り、注意された単語を可視化したHTMLファイルをvisualization/attentionフォルダに生成します。
* 全結果を20個ごとに1つのHTMLファイルに出力します。
* 注意結果に合わせて、ラベルと予測結果の候補を5件表示します。

## 入力データファイル

* 分類とテキストを記載したファイルです。
* データに応じて、以下のパラメータを調整してください。

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

* 入力データを形態素解析した結果に対して、類義語と一致する単語を見出し語に変換する辞書です。
* 見出し語を空文字にすることで、ストップワードを実現できます。

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

## ラベルデータファイル

* 分類番号に対するラベル名を記載したファイルです。
* 分類番号には行番号(0番号)が対応します。
* CSV形式のファイルで、各行の先頭の文字列がラベル名になります。
* 先頭以外の文字列は無視します。

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

