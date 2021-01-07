import torch

class DefaultLSTM(torch.nn.Module):

    def __init__(self, n_categories, embedding_size=100, hidden_size=25, drop_prob=0.2):
        super(DefaultLSTM, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_categories = n_categories

        self.lstm = torch.nn.LSTM(self.embedding_size, hidden_size=hidden_size, batch_first=True)
        self.drop = torch.nn.Dropout(p=drop_prob)
        self.out = torch.nn.Linear(self.hidden_size, self.n_categories)

    def forward(self, batch):

        lstm_out, h = self.lstm(batch)
        return self.out(self.drop(lstm_out))
