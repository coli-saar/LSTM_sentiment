import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import settings

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BaselineModel(nn.Module):

    def __init__(self, **kwargs):
        """
        Initialize new baseline model.
        :keyword argument: hidden_size: int, number of hitten units.
        """
        super().__init__()
        self.hidden_size = kwargs["hidden_size"]

        self.rnn = nn.RNN(input_size=256, hidden_size=self.hidden_size, num_layers=1)
        self.output_layer = nn.Linear(self.hidden_size, 4)
        self.h0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))

    def forward(self, sequence):
        sequence = sequence.permute(1, 0, 2)
        output, hn = self.rnn(sequence, self.h0)
        predictions = self.output_layer(output)
        return predictions

    def get_name(self):
        return "BaseLine_{}".format(self.hidden_size)


class PureGRU(nn.Module):

    def __init__(self, **kwargs):
        """
        Initialize new PureGRU model.
        :keyword arguments:
        hidden_size: int, number of hitten units.
        num_layers: int, Number of recurrent layers
        """
        super().__init__()
        self.hidden_size = kwargs["hidden_size"]
        self.num_layers = kwargs["num_layers"]

        self.gru = nn.GRU(input_size=256, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.output_layer = nn.Linear(self.hidden_size, 4)
        self.h0 = nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_size))

    def forward(self, sequence):
        sequence = sequence.permute(1, 0, 2)
        output, hn = self.gru(sequence, self.h0)
        predictions = self.output_layer(output)
        return predictions

    def get_name(self):
        return "PureGRU_h{}_l{}".format(self.hidden_size, self.num_layers)


class SimpleLSTM(nn.Module):

    def __init__(self, **kwargs):
        """
        Initialize new SimpleLSTM model.
        :keyword argument: hidden_size: int, number of hitten units.
        """
        super().__init__()
        self.hidden_size = kwargs["hidden_size"]

        self.lstm = nn.LSTM(input_size=256, hidden_size=self.hidden_size, num_layers=1)
        self.output_layer = nn.Linear(self.hidden_size, 4)
        self.h0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.c0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))

    def forward(self, sequence):
        sequence = sequence.permute(1, 0, 2)
        output, hn = self.lstm(sequence, (self.h0, self.c0))
        predictions = self.output_layer(output)
        return predictions

    def get_name(self):
        return "SimpleLSTM_{}".format(self.hidden_size)


class EmbeddingLSTM(nn.Module):
    """ LSTM Model that learns its own character embeddings. Max index 256 hardcoded. """

    def __init__(self, **kwargs):
        """
        Initalize new EmbeddingLSTM model.
        :keyword hidden_size: number of hidden units.
        :keyword num_layers: number of LSTM layers.
        :keyword embedding_dim: Dimensionality of the embeddings.
        """
        super().__init__()
        self.hidden_size = kwargs["hidden_size"]
        self.num_layers = kwargs["num_layers"]
        self.embedding_dim = kwargs["embedding_dim"]

        self.char_embeddings = nn.Embedding(256, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.output_layer = nn.Linear(self.hidden_size, 4)

        self.h0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.c0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))

    def forward(self, sequence):
        embeds = self.char_embeddings(sequence)
        output, hn = self.lstm(embeds.permute(1, 0, 2), (self.h0, self.c0))
        predictions = self.output_layer(output)
        return predictions

    def get_name(self):
        return "EmbeddingLSTM_h{}_l{}_e{}".format(self.hidden_size, self.num_layers, self.embedding_dim)


class EmbeddingBaselineModel(nn.Module):

    def __init__(self, **kwargs):
        """
        Initialize new baseline model.
        :keyword argument: hidden_size: int, number of hitten units.
        :keyword num_layers: number of LSTM layers.
        :keyword embedding_dim: Dimensionality of the embeddings.
        """
        super().__init__()
        self.hidden_size = kwargs["hidden_size"]
        self.num_layers = kwargs["num_layers"]
        self.embedding_dim = kwargs["embedding_dim"]

        self.char_embeddings = nn.Embedding(256, self.embedding_dim)
        self.rnn = nn.RNN(input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.output_layer = nn.Linear(self.hidden_size, 4)
        self.h0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))

    def forward(self, sequence):
        embeds = self.char_embeddings(sequence)
        sequence = embeds.permute(1, 0, 2)
        output, hn = self.rnn(sequence, self.h0)
        predictions = self.output_layer(output)
        return predictions

    def get_name(self):
        return "EmbeddingBaseLine_h{}_l{}_e{}".format(self.hidden_size, self.num_layers, self.embedding_dim)


class EmbeddingGRU(nn.Module):

    def __init__(self, **kwargs):
        """
        Initialize new PureGRU model.
        :keyword arguments:
        hidden_size: int, number of hitten units.
        num_layers: int, Number of recurrent layers
        :keyword embedding_dim: Dimensionality of the embeddings.
        """
        super().__init__()
        self.hidden_size = kwargs["hidden_size"]
        self.num_layers = kwargs["num_layers"]
        self.embedding_dim = kwargs["embedding_dim"]

        self.char_embeddings = nn.Embedding(256, self.embedding_dim)
        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.output_layer = nn.Linear(self.hidden_size, 4)
        self.h0 = nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_size))

    def forward(self, padded, lengths):
        #padded, lengths = pad_packed_sequence(sequence, padding_value=0)
        embeds = self.char_embeddings(padded)
        sequence = pack_padded_sequence(embeds, lengths)
        output, hn = self.gru(sequence, self.h0.expand(self.num_layers, len(lengths), self.hidden_size))
        predictions = self.output_layer(hn)
        return predictions

    def get_name(self):
        return "EmbeddingGRU_h{}_l{}_e{}".format(self.hidden_size, self.num_layers, self.embedding_dim)
