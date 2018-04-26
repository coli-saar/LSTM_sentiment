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

        self.name = "GroupModel(%s)_%d_%d" % (modelclass.__name__, num_users, num_groups)

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
        # likelihoods: np.array of shape [bs, K], where likelihoods[i,g] = P_g(yi|xi)
        # userids: [uid_1, ..., uid_bs]
        for i in range(len(likelihoods)):
            u = userids[i]
            self.likelihoods[u] *= likelihoods[i]

    def forward(self, userids, *original_inputs):
        predictions = [m(*original_inputs) for m in self.models]         # K x [bs, |Y|]
        prediction_matrix = torch.stack(predictions)                     # [K, bs, |Y|]
        prediction_matrix = torch.transpose(prediction_matrix, 0, 1)     # [bs, K, |Y|]
        prediction_matrix = torch.transpose(prediction_matrix, 1, 2)     # [bs, |Y|, K]

        group_probs = self.gu_posterior(userids)                         # [bs, K]
        group_probs = torch.unsqueeze(group_probs, 2)                    # [bs, K, 1]

        weighted_predictions = torch.bmm(prediction_matrix, group_probs) # [bs, |Y|, 1]
        weighted_predictions = torch.squeeze(weighted_predictions, 2)    # [bs, |Y|]

        return weighted_predictions, prediction_matrix.data.numpy()


    def get_name(self):
        return self.name


