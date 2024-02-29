
from pprint import pprint
from random import shuffle, seed
from math import floor
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
# batch size:
# sequence length:
# embedding dimension (d_model):
# vocab size:

BATCH_SIZE = 64

class PrepData:
    def __init__(
        self,
        train_file,
        eval_file,
    ) -> None:
        # 00. Train test split
        # 01. Read the data and tokenize
        self.train_en, self.train_cn = self.load_data(train_file)
        self.eval_en, self.eval_cn = self.load_data(eval_file)
        # 02. Build dictionary: en and cn
        self.en_word_index_map, self.en_index_word_map, self.en_vocab_size = self.build_dict(self.train_en)
        self.cn_word_index_map, self.cn_index_word_map, self.cn_vocab_size = self.build_dict(self.train_cn)
        # 03. word to id by dictionary Use input word list length to sort, reduce padding
        self.train_en, self.train_cn = self.word_to_id(self.train_en, self.train_cn, self.en_word_index_map, self.cn_word_index_map)
        self.eval_en, self.eval_cn = self.word_to_id(self.eval_en, self.eval_cn, self.en_word_index_map, self.cn_word_index_map)
        # 04. batch + padding + mask
        pass

    def build_dict(self, sentences: list[list[str]], max_words: int=5000):
        """build dictionary as {word: index}

        Args:
            sentences (list[list[str]]): _description_
            max_words (int, optional): _description_. Defaults to 5000.
        """
        word_counter = Counter()
        for sentence in sentences:
            for w in sentence:
                word_counter[w] += 1

        vocab = word_counter.most_common(max_words)
        vocab_size = len(vocab) + 2
        word_index_map = {w[0]: ind + 2 for ind, w in enumerate(vocab)}
        word_index_map['UNK'] = 0
        word_index_map['PAD'] = 1
        index_word_map = {ind: w for w, ind in word_index_map.items()}
        return word_index_map, index_word_map, vocab_size

    def word_to_id(
        self,
        en: list[list[str]],
        cn: list[list[str]],
        en_dict: dict,
        cn_dict: dict,
        sort: bool=True
    ):
        """convert input lists of words to lists of ids.

        Args:
            en (list[list[str]]): _description_
            cn (list[list[str]]): _description_
            en_dict (dict): _description_
            cn_dict (dict): _description_
            sort (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        n = len(en)
        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        def len_argsort(seq):
            return
        if sort:
            sorted_index = len_argsort(out_en_ids)
        return out_en_ids, out_cn_ids

    def split_batch(
        self,
        en: list[list[str]],
        cn: list[list[str]],
        batch_size: int,
        shuffle: bool = True
    ):
        idx_list = np.arange(0, len(en), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)

        return

    def load_raw_data(self, path, topk=10):
        """
        Read English and Chinese Data
        tokenize the sentence and add start/end marks(Begin of Sentence; End of Sentence)
        en = [['BOS', 'i', 'love', 'you', 'EOS'],
              ['BOS', 'me', 'too', 'EOS'], ...]
        cn = [['BOS', '我', '爱', '你', 'EOS'],
              ['BOS', '我', '也', '是', 'EOS'], ...]
        """
        data = []
        if topk is None:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(line)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = [next(f) for _ in range(topk)]
        return data

    def train_test_split(
            self,
            raw_data_path,
            train_path,
            test_path,
            topk=10
        ):
        seed(1)
        data = self.load_raw_data(raw_data_path, topk)
        shuffle(data)
        split = 0.7
        split_index = floor(len(data) * split)
        training = data[:split_index]
        testing = data[split_index:]
        with open(train_path, 'w') as f:
            for line in training:
                f.write(line)

        with open(test_path, "w") as f:
            for line in testing:
                f.write(line)


    def load_data(self, path):
        """_summary_

        Args:
            path (_type_): _description_

        Returns:
            _type_: _description_
        """
        en = []
        cn = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                en.append(["BOS"] + word_tokenize(line[0].lower()) + ["EOS"])
                cn.append(["BOS"] + word_tokenize(" ".join([w for w in line[1]])) + ["EOS"])
                # cn.append(" ".join([w for w in line[1]]))
        return en, cn

if __name__ == "__main__":
    dataloader = PrepData(train_file='data/train.txt', eval_file='data/test.txt')
    # path = "data/cmn.txt"
    # data = dataloader.load_raw_data(path, topk=100)
    # # pprint(data)
    # # print(len(data))
    # dataloader.train_test_split(
    #     raw_data_path=path,
    #     train_path="data/train.txt",
    #     test_path="data/test.txt",
    #     topk=100
    # )
    en, cn = dataloader.load_data("data/test.txt")
    w2i, i2w, vocab_size = dataloader.build_dict(sentences=cn[:2], max_words=100)
    # print(cn)
    print(cn[:2])
    print(w2i)
    print(i2w)