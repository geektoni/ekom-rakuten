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

    def _words_to_vec(self, sentence):
        return torch.FloatTensor([self.embeddings[word] for word in sentence])

    def __getitem__(self, index):
        correct_class = self.cat2x[self.dataset_Y[index]]
        result = torch.zeros(self.categories, dtype=torch.long)
        result[correct_class] = 1
        return self._words_to_vec(self.dataset_X[index]), result

    def __len__(self):
        return self.len

