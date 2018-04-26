import torch
from torch import FloatTensor, IntTensor, LongTensor, nn
from torch.autograd import Variable
from torch.nn import Embedding
import utils
import numpy as np

import settings
from models import PureGRU, EmbeddingGRU


def describe(label, x):
    print("%s: type=%s, size=%s" % (label, type(x), x.size()))

class GroupModel(nn.Module):
    def __init__(self, num_users, num_groups, modelclass, **kwargs):
        super().__init__()

        # set up K copies of the underlying model
        self.num_groups = num_groups
        self.models = [modelclass(**kwargs) for i in range(num_groups)]
        for i, m in enumerate(self.models):
            self.add_module("model%d" % i, m)

        # create matrix for P(g|u)
        self.np_group_for_user = np.zeros([num_users, num_groups], dtype=np.float32)
        self.p_group_for_user = Embedding(num_users, num_groups)
        self.p_group_for_user.weight.data = torch.from_numpy(self.np_group_for_user) # can now make changes to emb weights by changing numpy array
        self.p_group_for_user.weight.requires_grad = False

        # initialize P(g|u)
        for i in range(num_users):
            self.np_group_for_user[i] = np.random.dirichlet(np.ones(num_groups))

        print(self.p_group_for_user.weight)

        self.name = "GroupModel_%d_%d_%s" % (num_users, num_groups, str(modelclass))


    def forward(self, x, lengths, userids):
        predictions = [m(x, lengths) for m in self.models]   # K x [1, bs, |Y|]
        prediction_matrix = torch.stack(predictions)         # [K, 1, bs, |Y|]
        num_outputs = prediction_matrix.size()[-1]           # = |Y|
        prediction_matrix = prediction_matrix.view(self.num_groups, -1, num_outputs)  # [K, bs, |Y|]  # XXX
        prediction_matrix = torch.transpose(prediction_matrix, 0, 1)  # [bs, K, |Y|]
        prediction_matrix = torch.transpose(prediction_matrix, 1, 2)  # [bs, |Y|, K]
        # describe("prediction_matrix trans2", prediction_matrix)

        group_probs = self.p_group_for_user(userids)        # [bs, K]
        group_probs = torch.unsqueeze(group_probs, 2) # [bs, K, 1]
        # describe("gp", group_probs)

        weighted_predictions = torch.bmm(prediction_matrix, group_probs) # [bs, |Y|, 1]
        weighted_predictions = torch.squeeze(weighted_predictions, 2)    # [bs, |Y|]

        weighted_predictions = torch.unsqueeze(weighted_predictions, 1) # [bs, 1, |Y|]  # XXX
        # describe("wp", weighted_predictions)

        return weighted_predictions

    def get_name(self):
        return self.name


#
#
# gm = GroupModel(3, 2, EmbeddingGRU, hidden_size=5, num_layers=1, embedding_dim=2)
#
# xx = [Variable(LongTensor([1, 2])), Variable(LongTensor([3,4]))]
# features = utils.pack_sequence(xx)
# x, lengths = torch.nn.utils.rnn.pad_packed_sequence(features, padding_value=0)
#
# userids = torch.LongTensor([1, 1])
#
# pred = gm(x, lengths, userids)
#
# print(pred.size())
# print(pred)