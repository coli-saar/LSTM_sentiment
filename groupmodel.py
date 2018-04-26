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

        self.num_groups = num_groups
        self.num_users = num_users

        # set up K copies of the underlying model
        self.models = [modelclass(**kwargs) for i in range(num_groups)]
        for i, m in enumerate(self.models):
            self.add_module("model%d" % i, m)

        # create matrix for posteriors P(g|D_u)
        self.np_gu_posterior = np.zeros([num_users, num_groups], dtype=np.float32)
        self.gu_posterior = Embedding(num_users, num_groups)
        self.gu_posterior.weight.data = torch.from_numpy(self.np_gu_posterior) # can now make changes to emb weights by changing numpy array
        self.gu_posterior.weight.requires_grad = False

        # initialize posteriors P(g|D_u)
        for i in range(num_users):
            self.np_gu_posterior[i] = np.random.dirichlet(np.ones(num_groups))

        # set up data structure for priors P(g)
        self.np_g_prior = np.zeros(num_groups, dtype=np.float32)

        self.name = "GroupModel_%d_%d_%s" % (num_users, num_groups, str(modelclass))

    # call at the beginning of each training epoch
    def start_epoch(self):
        self.recalculate_priors()
        self.reset_likelihoods()

    # call at the end of each training epoch
    def end_epoch(self):
        self.recalculate_posteriors()

    def recalculate_priors(self):
        self.np_g_prior = self.np_gu_posterior.sum(axis=0) / self.num_users

    def recalculate_posteriors(self):
        # calculate P(g, D_u)
        joint = self.np_g_prior * self.likelihoods # (num_users, num_groups)
        Z = joint.sum(axis=1) # (num_users)
        self.np_gu_posterior = (joint.transpose() / Z).transpose() # (num_users, num_groups)

    def reset_likelihoods(self):
        self.likelihoods = np.ones([self.num_users, self.num_groups], dtype=np.float32)

    # call after each call to forward during training
    def collect_likelihoods(self, likelihoods, userids):
        # likelihoods: np.array of shape [bs, G], where likelihoods[i,g] = P_g(yi|xi)
        # userids: [uid_1, ..., uid_bs]
        for i in range(len(likelihoods)):
            u = userids[i]
            self.likelihoods[u] *= likelihoods[i]

    def forward(self, x, lengths, userids):
        predictions = [m(x, lengths) for m in self.models]   # K x [1, bs, |Y|]
        prediction_matrix = torch.stack(predictions)         # [K, 1, bs, |Y|]
        num_outputs = prediction_matrix.size()[-1]           # = |Y|
        prediction_matrix = prediction_matrix.view(self.num_groups, -1, num_outputs)  # [K, bs, |Y|]  # XXX
        prediction_matrix = torch.transpose(prediction_matrix, 0, 1)  # [bs, K, |Y|]
        prediction_matrix = torch.transpose(prediction_matrix, 1, 2)  # [bs, |Y|, K]
        # describe("prediction_matrix trans2", prediction_matrix)

        group_probs = self.gu_posterior(userids)        # [bs, K]
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