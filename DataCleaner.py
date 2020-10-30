import torch
import pandas as pd
from re import sub
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation


class QuoraDataCleaner:
    stopwords: list
    word_index: dict
    num_words: int
    vocab_size: int
    padding: str
    truncate: str

    def __init__(self, dataset, vocab_size, padding, truncate, oov, max_length):
        self.stopwords = stopwords.words('english')
        self.stopwords.extend(list(punctuation))
        self.padding = padding
        self.truncate = truncate
        self.vocab_size = vocab_size - 1
        self.oov = oov
        self.max_len = max_length
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.vocabulary = Dictionary(self.vocab_size, self.oov)
        self.longest_sentence = 0
        self.dataset = dataset

    def register(self, sentence):
        sentence = sentence.lower()
        sentence = sub(r"can[’']t", 'can not', sentence)
        sentence = sub(r"won[’']t", "will not", sentence)
        sentence = sub(r"shan[’']t", "shall not", sentence)
        sentence = sub(r"[’']s", ' is', sentence)
        sentence = sub(r"n[’']t", ' not', sentence)
        sentence = sub(r"i[’']m", 'i am', sentence)
        sentence = sub(r"[’']ve", ' have', sentence)
        sentence = sub(r"[’']re", " are", sentence)
        sentence = sub(r"[’']d", ' would', sentence)
        sentence = sub(r"[’']ll", ' will', sentence)
        try:
            words = [self.vocabulary.add_word(word)[0] for word in word_tokenize(sentence) if word not in
                     self.stopwords]
            self.longest_sentence = len(words) if self.longest_sentence < len(words) else \
                self.longest_sentence
            return ' '.join(words)
        except AttributeError as err:
            print("{}\nsentence: {}".format(err, sentence))
        return sentence

    def convert_to_seqs(self, row):
        try:
            cols = ['question1', 'question2']
            for col in cols:
                questions = row[col]
                words = word_tokenize(questions)
                question_seq = [self.vocabulary.word2idx.get(word, 0) for word in words]
                seq_len = len(question_seq)
                if seq_len < self.max_len:
                    if self.padding is 'left':
                        question_seq = [0] * (self.max_len - seq_len) + question_seq
                    elif self.padding is 'right':
                        question_seq = question_seq + [0] * (self.max_len - seq_len)
                elif seq_len > self.max_len:
                    if self.truncate is 'left':
                        question_seq = question_seq[-self.max_len:]
                    elif self.truncate is 'right':
                        question_seq = question_seq[:self.max_len]
                row[col] = question_seq
            return row
        except TypeError as t:
            print(t)
            print(row)

    def fit(self, data):
        data['question1'] = data['question1'].apply(self.register)
        data['question2'] = data['question2'].apply(self.register)
        self.vocabulary.create_vocab()
        print("fit done")
        return self._transform(data)

    def _transform(self, data):
        X = data[['question1', 'question2']].copy()
        X = X.apply(self.convert_to_seqs, axis=1)
        print("transform done")
        X = pd.concat([X, data['is_duplicate']], axis=1)
        return X

    def transform(self, data):
        return self._transform(data)


class Dictionary(object):
    def __init__(self, vocab_size, oov_token='<UNK>'):
        self.word2idx = dict()
        self.idx2word = dict()
        self.vocab_size = vocab_size
        self.oov = oov_token

    def add_word(self, word):
        if word not in self.word2idx.keys():
            self.word2idx[word] = 1
        self.word2idx[word] += 1
        return word, self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def create_vocab(self):
        self.word2idx = dict(sorted(self.word2idx.items(), key=lambda item: item[1], reverse=True)[:self.vocab_size])
        self.idx2word = dict([(i, word) for i, word in enumerate(self.word2idx.keys(), 2)])
        self.word2idx = {value: key for key, value in self.idx2word.items()}
        self.word2idx[self.oov] = 1
        self.idx2word[1] = self.oov
        print(len(self.word2idx))
        print(len(self.idx2word))


class QuoraDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, vocab_size=10000, padding='left', truncate='left', oov='<UNK>', max_length=100):
        super(QuoraDataset).__init__()
        data = dataset.dropna(axis=0)
        data = data.reset_index(drop=True)
        cleaner = QuoraDataCleaner(data, vocab_size, padding, truncate, oov, max_length)
        self.dataset = cleaner.fit(data)
        self.cleaner = cleaner

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # print(idx)
        return self.dataset.iloc[idx]


def generate_batch(batch):
    # print("generate batch")
    try:
        q1 = torch.tensor([item['question1'] for item in batch])
        q2 = torch.tensor([item['question2'] for item in batch])
        label = torch.tensor([item['is_duplicate'] for item in batch])
        # print(q1.size(), q2.size(), label.size())
        return q1, q2, label
    except ValueError as v:
        print(v)