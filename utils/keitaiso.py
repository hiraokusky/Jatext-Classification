import codecs
import unicodedata
import neologdn
import re
import csv

import janome.tokenizer
from distribute import JDistribution

janomet = janome.tokenizer.Tokenizer(mmap=True)

class JKeitaiso:
    """
    テキストを形態素解析してトークン列にする
    """

    # 対象とするテキスト数
    num = 10
    # 対象外とするテキストサイズ
    min_len = 200

    def load_from_file(self, data_path):
        """
        ファイルからラベルとテキスト情報を取り出す
        """
        full_dataset = []
        with open(data_path, 'r', encoding="cp932") as f:
            reader = csv.reader(f)
            i = 0
            for line in reader:
                text = line[1]
                if len(line) > 0 and len(text) > self.min_len:
                    full_dataset.append([line[0], text])
                else:
                    print(len(text))
                i += 1

                # if i >= self.num:
                #     break

        return full_dataset

    def load_corpus_from_file(self, corpus_path):
        full_dataset = []
        with open(corpus_path, 'r', encoding="utf8") as f:
            reader = csv.reader(f)
            i = 0
            for line in reader:
                if len(line) > 0:
                    full_dataset.append(['0', line[0]])
                i += 1
        return full_dataset

    def get_soap(self, line):
        """
        テキストの構造を分析して、必要な情報だけを取り出す
        """
        p = ''
        q = ''
        mode = ''
        d = {'Ｓ':'', 'Ｏ':'', 'Ａ':'', 'Ｐ':'', 'P':'', 'Ｑ':'', 'Ｒ':'', 'S':'', 'Ｔ':''}
        for c in line:
            if c in ['Ｓ', 'Ｏ', 'Ａ', 'Ｐ', 'Ｑ', 'Ｒ', 'Ｔ']:
                if c in ['Ｏ']: # OPQRSTの構造で始まりを判断
                    mode = 'O'
                elif c in ['Ｑ', 'Ｒ']: # SOAPの構造か、OPQRSTの構造で終わりを判断
                    mode = 'R'
                elif c in ['Ａ', 'Ｔ']: # SOAPの構造か、OPQRSTの構造で終わりを判断
                    mode = ''

                if mode == 'O' and c == 'Ｐ': # OPQRSTのPであることを判断
                    p = 'P'
                if mode == 'R' and c == 'Ｓ': # OPQRSTのPであることを判断
                    p = 'S'
                else:
                    p = c
            elif p != '' and (c == '：' or c == ' '):
                q = p
                d[q] += ' '
            elif q != '':
                d[q] += c
        so = 'S ' + d['Ｓ'] + ' O ' + d['Ｏ'] + ' P ' + d['P'] + ' Q ' + d['Ｑ'] + ' R ' + d['Ｒ'] + ' S ' + d['S'] + ' T ' + d['Ｔ']
        return so

    def parse_structure(self, line):
        """
        テキストの構造を分析して、必要な情報だけを取り出す
        """
        dst = self.get_soap(line)
        return dst

    # 句読点と空白を区切り文字とする
    tt_seps = str.maketrans('、。 　', ' ...')

    tt_ksuji = str.maketrans('一二三四五六七八九〇壱弐参', '1234567890123')
    re_suji = re.compile(r'[十拾百千万億兆\d]+')
    re_kunit = re.compile(r'[十拾百千]|\d+')
    re_manshin = re.compile(r'[万億兆]|[^万億兆]+')

    TRANSUNIT = {'十': 10,
                '拾': 10,
                '百': 100,
                '千': 1000}
    TRANSMANS = {'万': 10000,
                '億': 100000000,
                '兆': 1000000000000}

    def kansuji2arabic(self, kstring: str, sep=False):
        """
        漢数字をアラビア数字に変換する
        """
        def _transvalue(sj: str, re_obj=self.re_kunit, transdic=self.TRANSUNIT):
            unit = 1
            result = 0
            for piece in reversed(re_obj.findall(sj)):
                if piece in transdic:
                    if unit > 1:
                        result += unit
                    unit = transdic[piece]
                else:
                    val = int(piece) if piece.isdecimal() else _transvalue(piece)
                    result += val * unit
                    unit = 1

            if unit > 1:
                result += unit

            return result

        transuji = kstring.translate(self.tt_ksuji)
        for suji in sorted(set(self.re_suji.findall(transuji)), key=lambda s: len(s),
                            reverse=True):
            if not suji.isdecimal():
                arabic = _transvalue(suji, self.re_manshin, self.TRANSMANS)
                arabic = '{:,}'.format(arabic) if sep else str(arabic)
                transuji = transuji.replace(suji, arabic)
            elif sep and len(suji) > 3:
                transuji = transuji.replace(suji, '{:,}'.format(int(suji)))

        return transuji

    def normalize(self, word):
        """
        テキストを正規化する
        """
        # 前後空白を削除
        word = word.strip()
        # 日本語の区切りをシンプルに変換
        word = word.translate(self.tt_seps)
        # 小文字化
        word = word.lower()
        # 漢数字をアラビア数字にする
        word = self.kansuji2arabic(word)
        # NFKC（Normalization Form Compatibility Composition）で
        # 半角カタカナ、全角英数、ローマ数字・丸数字、異体字などなどを正規化。
        word = unicodedata.normalize("NFKC", word)
        # アルファベットやアラビア数字、括弧やエクスクラメーションマークなどの記号は、半角に統一
        # カタカナは、全角に統一
        # "うまーーーい!!" → "うまーい!" など、重ね表現を除去
        # 一方で、"やばっっっ!!" は除去できてません
        # repeat引数に1を渡すと、2文字以上の重ね表現は1文字にできますが、そうすると"Good"は"God"になってしまったりします
        # ”〜”などの記号も除去されてます
        word = neologdn.normalize(word)

        # もろもろ正規化したあとのパターンマッチング変換

        # URLを削除
        word = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', word)
        # 桁区切りの除去と数字の置換
        word = re.sub(r'(\d)([,.])(\d+)', r'\1\3', word)
        word = re.sub(r'\d+', '0', word)
        # 記号削除
        word = re.sub(r'[\(\)\<\>\[\]\【\】\《\》\≪\≫\/\#\?\・]', '', word)

        return word

    novalues = [
        'しない',
        'ない',
        '無し',
        'なし',
        'ない',
        '陰性',
        '低い',
        '低下',
        '不能',
        '-/-',
        '-',
        '(-/-)',
        '(-)',
    ]

    def is_negative(self, w):
        """
        否定表現かどうか判定する
        """
        if w in self.novalues:
            return True
        return False

    def get_tokens(self, line, synonyms=[]):
        """
        テキストをトークン化する
        """
        tokens = []
        line = self.normalize(line)
        # 改行で分割する
        lines = line.split('.')
        for line in lines:
            if len(line) > 0:
                line += ' .'
                for node in janomet.tokenize(line):
                    parts = node.part_of_speech.split(',')
                    part = parts[0]
                    t = ''
                    # print(node)
                    if self.is_negative(node.surface):
                        t = 'ない'
                    elif self.is_negative(node.base_form):
                        t = 'ない'
                    elif node.surface in ['s','o','p','q','r','t','.']:
                        t = node.surface
                    elif parts[1] in ['記号']:
                        t = ''
                    elif parts[1] in ['数']:
                        t = '<UNK>'
                    elif parts[1] in ['固有名詞']:
                        t = '<UNK>'
                        # t = node.surface
                    elif part in ['名詞']:
                        t = self.match_syns(node.surface, synonyms)
                    elif part in ['動詞', '形容詞', '副詞']:
                        t = self.match_syns(node.base_form, synonyms)
                    # else:
                    #     t = node.surface
                    
                    if len(t) > 0:
                        tokens.append(t)
        
        return tokens

    def load_synonym_dict(self, dictname):
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

    def match_syns(self, s, synonyms=[]):
        """
        類義語とストップワードの処理
        """
        if len(s) > 0 and len(synonyms) > 0:
            for synonyms1 in synonyms:
                cat = synonyms1[0]
                synonyms1_key = synonyms1[1]
                synonyms1_val = synonyms1[2]
                if s in synonyms1_val:
                    return synonyms1_key
        return s

    # コーパスをつくるときにはTrueにする
    # 学習データをパースするときにはFalseにする
    # TrueだとSOAP解析をしない
    def run(self, data, path_w, synosyms, is_corpus = False, start = 0, end = 0):
        """
        データを形態素解析してファイルに保存する
        """
        import datetime
        now = datetime.datetime.now()
        i = 0
        lines = []
        for line in data:
            if end > 0 and (i < start or i >= end):
                break

            if len(data) < 1000:
                print(i + 1, '/', len(data))
            line1 = line[1]
            if not is_corpus:
                line1 = self.parse_structure(line1)
            line1 = self.get_tokens(line1, synosyms)
            lines.append([line[0], line1])

            i += 1

            if i % 1000 == 0:
                t = datetime.datetime.now() - now
                r = len(data) - i
                print(i, '/', len(data), 'Remaining:', r / 1000 * t)
                now = datetime.datetime.now()

            # if i > 3000:
            #     break

            # line1 = ' '.join(line1)
            # print(line1)

        with open(path_w, mode='w',encoding='cp932', errors='ignore') as f:
            for item in lines:
                line1 = ' '.join(item[1])
                print(item[0] + ',' + line1, file=f)

    # # 文字列配列を1行のcsv形式に変換する
    # def array_to_csv_line(self, label, array_word):
    #     csv_line = label
    #     for csv_word in array_word:
    #         if len(csv_line) > 0:
    #             csv_line += ','
    #         csv_line += csv_word
    #     return csv_line

jk = JKeitaiso()
synosyms = jk.load_synonym_dict('db/dict.csv')

path_w = "db/token.csv"
data = jk.load_from_file('db/all.csv')

jk.run(data, path_w, synosyms)

path_w = "db/corpus"
data2 = jk.load_corpus_from_file('db/bccwj.core')
data.extend(data2)
start = 0
end = 100000
path_w += str(start) + '-' + str(end) + '.csv'

jk.run(data, path_w, synosyms, True, start, end)

jd = JDistribution()
jd.create_model(path_w)

# 類義語チェックテスト
results = jd.get_synonyms('めまい')
for result in results:
    print(result)

# jd.create_dictionary('db/words.txt', 'db/token.csv')
# embeddings = jd.load_embeddings('db/words.txt')
