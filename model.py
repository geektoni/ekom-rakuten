import torch


class BidLSTM(torch.nn.Module):
    """
    Bidirectional LSTM used for this task
    """

    def __init__(self, n_categories, embedding_size=100, hidden_size=25, drop_prob=0.2):
        """
        Initialize the model
        :param n_categories: total categories available (length of the final prediction)
        :param embedding_size: size of the embedding vectors
        :param hidden_size: size of the hidden layer
        :param drop_prob: dropout probability
        """
        super(BidLSTM, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # We increase the category size to account for unknown categories
        # which are not present in the training set.
        self.n_categories = n_categories+1

        self.lstm = torch.nn.LSTM(self.embedding_size, hidden_size=hidden_size,
                                  batch_first=True, bidirectional=True)
        self.drop = torch.nn.Dropout(p=drop_prob)
        self.out = torch.nn.Linear(2*self.hidden_size, self.n_categories)
        self.relu = torch.nn.ReLU()

    def forward(self, batch):

        lstm_out, h = self.lstm(batch)
        return self.relu(self.out(self.drop(lstm_out[:, -1, :])))
