import sys
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
        self.input_size = kwargs["input_size"]
        self.num_layers = kwargs["num_layers"]

        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.output_layer = nn.Linear(self.hidden_size, 4)
        #self.h0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        self.float_tensor = torch.cuda.FloatTensor if settings.GPU else torch.FloatTensor

    def forward(self, padded, lengths):
        sequence = pack_padded_sequence(padded, lengths)
        h0 = Variable(self.float_tensor(self.num_layers, len(lengths), self.hidden_size).fill_(0.))
        output, hn = self.rnn(sequence, h0)
        predictions = self.output_layer(hn)
        return predictions

    def get_name(self):
        return "BaseLine_h{}_l{}_i{}".format(self.hidden_size, self.num_layers, self.input_size)


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
        self.input_size = kwargs["input_size"]

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.output_layer = nn.Linear(self.hidden_size, 4)
        #self.h0 = nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_size))

        self.float_tensor = torch.cuda.FloatTensor if settings.GPU else torch.FloatTensor

    def forward(self, padded, lengths):
        sequence = pack_padded_sequence(padded, lengths)
        h0 = Variable(self.float_tensor(self.num_layers, len(lengths), self.hidden_size).fill_(0.))
        output, hn = self.gru(sequence, h0)
        predictions = self.output_layer(hn)
        return predictions

    def get_name(self):
        return "PureGRU_h{}_l{}_i{}".format(self.hidden_size, self.num_layers, self.input_size)


# AK
# = PureGRU + log-softmax layer
class PureGRUClassifier(nn.Module):

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
        self.input_size = kwargs["input_size"]

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.pre_output_layer = nn.Linear(self.hidden_size, 2)
        self.output_layer = nn.LogSoftmax(dim=-1) # normalize along last dimension

        # self.output_layer = nn.Linear(self.hidden_size, 4)
        #self.h0 = nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_size))

        self.float_tensor = torch.cuda.FloatTensor if settings.GPU else torch.FloatTensor

    def forward(self, padded, lengths):
        sequence = pack_padded_sequence(padded, lengths)
        h0 = Variable(self.float_tensor(self.num_layers, len(lengths), self.hidden_size).fill_(0.))
        output, hn = self.gru(sequence, h0)
        pre = self.pre_output_layer(hn)
        # print("pre %s" % str(pre.size()))
        predictions = self.output_layer(pre)
        return predictions.squeeze(0) # return [bs, |Y|] instead of [1, bs, |Y|]

    def get_name(self):
        return "PureGRU_h{}_l{}_i{}".format(self.hidden_size, self.num_layers, self.input_size)

class SimpleLSTM(nn.Module):

    def __init__(self, **kwargs):
        """
        Initialize new SimpleLSTM model.
        :keyword argument: hidden_size: int, number of hitten units.
        """
        super().__init__()
        self.hidden_size = kwargs["hidden_size"]
        self.input_size = kwargs["input_size"]
        self.num_layers = kwargs["num_layers"]

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.output_layer = nn.Linear(self.hidden_size, 4)
        #self.h0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        #self.c0 = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        self.float_tensor = torch.cuda.FloatTensor if settings.GPU else torch.FloatTensor

    def forward(self, padded, lengths):
        sequence = pack_padded_sequence(padded, lengths)
        h0 = Variable(self.float_tensor(self.num_layers, len(lengths), self.hidden_size).fill_(0.))
        c0 = Variable(self.float_tensor(self.num_layers, len(lengths), self.hidden_size).fill_(0.))
        output, (hn, cn) = self.lstm(sequence, (h0, c0))
        predictions = self.output_layer(hn)
        return predictions

    def get_name(self):
        return "SimpleLSTM_h{}_l{}_i{}".format(self.hidden_size, self.num_layers, self.input_size)


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
        return "EmbeddingLSTM_h{}_l{}_i{}_e{}".format(self.hidden_size, self.num_layers, self.input_size, self.embedding_dim)


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
        #self.h0 = nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_size))

        self.float_tensor = torch.cuda.FloatTensor if settings.GPU else torch.FloatTensor

    def forward(self, padded, lengths):
        #padded, lengths = pad_packed_sequence(sequence, padding_value=0)
        embeds = self.char_embeddings(padded)
        sequence = pack_padded_sequence(embeds, lengths)
        h0 = Variable(self.float_tensor(self.num_layers, len(lengths), self.hidden_size).fill_(0.))
        output, hn = self.gru(sequence, h0)
        predictions = self.output_layer(hn)
        return predictions

    def get_name(self):
        return "EmbeddingGRU_h{}_l{}_e{}".format(self.hidden_size, self.num_layers, self.embedding_dim)



# AKAKAK
# A classifier variant of EmbeddingGRU, which predicts whether the review was positive (1) or negative (0).
class EmbeddingGRUClassifier(nn.Module):
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
        self.pre_output_layer = nn.Linear(self.hidden_size, 2)
        self.output_layer = nn.LogSoftmax()

        self.float_tensor = torch.cuda.FloatTensor if settings.GPU else torch.FloatTensor

    def forward(self, padded, lengths):
        #padded, lengths = pad_packed_sequence(sequence, padding_value=0)
        #AKAKAK
        print("padded: %s %s" % (type(padded), str(padded.size())))
        embeds = self.char_embeddings(padded)
        sequence = pack_padded_sequence(embeds, lengths)
        h0 = Variable(self.float_tensor(self.num_layers, len(lengths), self.hidden_size).fill_(0.))
        output, hn = self.gru(sequence, h0)
        predictions = self.output_layer(self.pre_output_layer(hn))
        return predictions

    def get_name(self):
        return "EmbeddingGRU_C_h{}_l{}_e{}".format(self.hidden_size, self.num_layers, self.embedding_dim)


class ConvLSTM(nn.Module):

    def __init__(self, **kwargs):
        """
        Initalize new ConvLSTM model.
        :keyword hidden_size: number of hidden units.
        :keyword num_layers: number of LSTM layers.
        :keyword embedding_dim: Dimensionality of the embeddings.
        :keyword kernel_size: Size of the kernel we use
        :keyword kernel_dim : Dimension of the kernel
        :keyword batch_size : Size of the batch
        """
        super().__init__()
        self.num_layers = kwargs["num_layers"]
        self.hidden_size = kwargs["hidden_size"]
        self.input_size = kwargs["input_size"]
        self.embedding_dim = kwargs["embedding_dim"]
        self.kernel_size = kwargs["kernel_size"]
        self.cnn_padding = int((self.kernel_size - 1) / 2)
        self.intermediate_size = kwargs["intermediate_size"]
        self.dropout = kwargs["dropout"]
        self.char_embeddings = nn.Embedding(256, self.embedding_dim)

        # Only accept odd kernel sizes
        if self.kernel_size % 2 == 0:
            raise AttributeError("Only odd kernel sizes accepted for CNN-LSTM")
     
        # CNN
        #self.conv = nn.Conv2d(1, self.kernel_nb, (self.kernel_size, 256))
        self.conv = nn.Conv1d(self.input_size, self.intermediate_size, self.kernel_size, padding=self.cnn_padding)
        self.conv_actiation = F.relu
        self.dropout = nn.Dropout(self.dropout)

        # LSTM
        self.lstm = nn.LSTM(self.intermediate_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.output_layer = nn.Linear(self.hidden_size, 4)

        self.float_tensor = torch.cuda.FloatTensor if settings.GPU else torch.FloatTensor

    def forward(self, padded, lengths):
        #CNN
        permuted = padded.permute(1, 2, 0)
        intermediate = self.dropout(self.conv_actiation(self.conv(permuted)))
        intermediate = intermediate.permute(2, 0, 1)
        #LSTM
        sequence = pack_padded_sequence(intermediate, lengths)
        h0 = Variable(self.float_tensor(self.num_layers, len(lengths), self.hidden_size).fill_(0.))
        c0 = Variable(self.float_tensor(self.num_layers, len(lengths), self.hidden_size).fill_(0.))
        output, (hn, cn) = self.lstm(sequence, (h0, c0))
        predictions = self.output_layer(hn)
        return predictions

    def get_name(self):
        return "ConvLSTM_h{}_l{}_i{}_int{}".format(self.hidden_size, self.num_layers, self.input_size, self.intermediate_size)

