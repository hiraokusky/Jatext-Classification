# word2vecを利用した機械学習

## ファイルを形態素に分解する

python keitaiso.py

類似語辞書をロードする
db/dict.csv

データをロードし、形態素解析して、token.csvを出力する
db/all.csv
→db/token.csv

コーパスをロードし、形態素解析して、corpus.csvを出力する
分割して出力できる
db/all.csv
db/bccwj.core
→db/corpus.csv

データとコーパスの形態素解析結果をロードし、word2vecモデルを作成する(distribute.py)
db/token.csv
db/corpus.csv
→db/corpus.model

## 学習する

python .\train.py -i db/token.csv -s db\dict.csv -l db\disease.csv

データの形態素解析結果をロードし、学習データとテストデータに分割し、単語辞書を作成し、単語をIDに置き換えて、単文に分割する
db/token.csv
→db/dict.txt

word2vecモデルをロードし、単語辞書とマッピングしたembeddingsを作成する
db/corpus.model
db/dict.txt

学習する

attentionを可視化したHTMLを作成する
