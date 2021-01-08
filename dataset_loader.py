import torch
from torch.utils.data import Dataset

import numpy as np

class RakutenLoader(Dataset):
    """
    Load the Rakuten dataset and make it available for batching.
    """

    def __init__(self, dataset_X, dataset_Y, embeddings, categories,
                 pad_sentence_length = 10, embeddings_length=100):
        """
        Init the loader
        :param dataset_X: product's titles
        :param dataset_Y: product's categories
        :param embeddings: embeddings used for converting the titles
        :param categories: dictionary which maps categories to integers
        :param pad_sentence_length: product title length (if lower we add padding)
        """

        self.embeddings = embeddings
        self.embeddings_dim = embeddings.get_dimension()

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
        """
        Given a sentence, we pad it with the symbol @PAD@ if its length
        is less than the target one. If the length is greater than the
        maximum, we trim the sentence.
        :param sentence: product's title
        :return: sentence padded or trimmed
        """
        if len(sentence) > self.pad_sentence_length:
            sentence = sentence[:self.pad_sentence_length]
            sentence[self.pad_sentence_length-1] = "@EOS@"
        elif len(sentence) < self.pad_sentence_length:
            sentence.extend(["@PAD@"] * (self.pad_sentence_length - len(sentence)))
        return sentence

    def _use_embeddings(self, word):
        """
        Given a word, we return its embedding. If the word is @PAD@ we
        return an empty embedding.
        :param word: target word
        :return: embedding as a numpy vector
        """
        if word == "@PAD@":
            return torch.zeros(self.embeddings_dim)
        else:
            return self.embeddings[word]

    def _words_to_vec(self, sentence):
        """
        Convert each word of a sentence into an embedding
        :param sentence: target sentence
        :return: embeddings matrix
        """
        return torch.FloatTensor([self._use_embeddings(word) for word in sentence])

    def _cat_to_idx(self, category):
        """
        Given a category we return its mapping into an id
        :param category: target category
        :return: id corresponding to that category
        """
        if not category in self.cat2x:
            return torch.LongTensor([self.categories])
        else:
            return torch.LongTensor([self.cat2x[category]])

    def __getitem__(self, index):
        return self._words_to_vec(self.dataset_X[index]), self._cat_to_idx(self.dataset_Y[index])

    def __len__(self):
        return self.len
