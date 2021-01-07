import torch
from torch.utils.data import Dataset

import numpy as np

class RakutenLoader(Dataset):

    def __init__(self, dataset_X, dataset_Y, embeddings, categories,
                 pad_sentence_length = 10):

        self.embeddings = embeddings

        self.pad_sentence_length=pad_sentence_length

        self.dataset_X = dataset_X.to_numpy()
        self.dataset_Y = dataset_Y.to_numpy()
        self.cat2x = categories
        self.categories = len(self.cat2x)

        self.len=len(dataset_X)

        # Pad the sentences so to have similar length
        self.dataset_X = np.array(list(
            map(lambda x: self._pad_sentence_length(x), self.dataset_X)
        ))

    def _pad_sentence_length(self, sentence):
        if len(sentence) > self.pad_sentence_length:
            sentence = sentence[:self.pad_sentence_length]
            sentence[self.pad_sentence_length-1] = "@EOS@"
        elif len(sentence) < self.pad_sentence_length:
            sentence.extend(["@PAD@"] * (self.pad_sentence_length - len(sentence)))
        return sentence

    def _use_embeddings(self, word):
        if word == "@PAD@":
            return torch.zeros(100)
        else:
            return self.embeddings[word]

    def _words_to_vec(self, sentence):
        return torch.FloatTensor([self._use_embeddings(word) for word in sentence])

    def _cat_to_idx(self, category):
        if not category in self.cat2x:
            return torch.LongTensor([self.categories])
        else:
            return torch.LongTensor([self.cat2x[category]])

    def __getitem__(self, index):
        return self._words_to_vec(self.dataset_X[index]), self._cat_to_idx(self.dataset_Y[index])

    def __len__(self):
        return self.len
