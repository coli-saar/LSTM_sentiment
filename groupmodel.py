import sys

import time
import torch
from torch import FloatTensor, IntTensor, LongTensor, nn
from torch.autograd import Variable
from torch.nn import Embedding
import utils
import numpy as np
from collections import defaultdict

import settings
from models import PureGRU, EmbeddingGRU


def describe(label, x):
    print("%s: type=%s, size=%s" % (label, type(x), x.size()))

class GroupModel(nn.Module):
    def __init__(self, num_users, num_groups, models): # , modelclass
        super().__init__()

        self.num_groups = num_groups
        self.num_users = num_users

        # make parameters of the individual models parameters of the group model
        # self.models = [modelclass(**kwargs) for i in range(num_groups)]
        for i, m in enumerate(models):
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

        # set model name
        # underlying_model_name = self.models[0].get_name()

        # self.rnnweights = [m.state_dict()["gru.weight_hh_l0"] for m in self.models]

        # self.weights0 = self.models[0].state_dict()["gru.weight_hh_l0"]
        # self.weights1 = self.models[1].state_dict()["gru.weight_hh_l0"]

        self.name = "GroupModel(%d)" % (num_groups) # , underlying_model_name)

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
        nll = self.nll.cpu().numpy()
        likelihoods = np.exp(-nll)  # (U,K)

        joint = self.np_g_prior * likelihoods # np.exp(self.log_likelihoods) # (num_users, num_groups)
        Z = joint.sum(axis=1) # (num_users)
        self.np_gu_posterior = (joint.transpose() / Z).transpose() # (num_users, num_groups)

    def mean_posterior_entropy(self):
        s = 0
        for i in range(self.num_users):
            pk = self.np_gu_posterior[i]
            s += -np.sum(pk * np.log2(pk), axis=0)
        return (s/self.num_users)

    # def cosine_distance(self):
    #     if self.num_groups > 1:
    #         return self.cos(self.rnnweights[0].numpy(), self.rnnweights[1].numpy())
    #     else:
    #         return 0

    def cos(self, v1, v2):
        v1 = v1.ravel()
        v2 = v2.ravel()
        ret = (v1.dot(v2)) / np.sqrt(v1.dot(v1)) / np.sqrt(v2.dot(v2))
        print("cos: %f" % ret)
        return ret

    def vecdiff(self, v1, v2):
        v1, v2 = v1.ravel(), v2.ravel()
        ret = np.sum(np.abs(v1-v2))/len(v1)
        # print("diff %f" % ret)
        return ret

    def reset_likelihoods(self):
        self.nll = settings.cd(torch.zeros((self.num_users, self.num_groups))) # (U,K)

        # self.log_likelihoods = np.zeros([self.num_users, self.num_groups], dtype=np.float32)
        # self.sumcos = 0
        # self.Z_sumcos = 0

    # call after each call to forward during training
    # DEPRECATED
    def collect_log_likelihoods(self, log_likelihoods, userids):
        # likelihoods: np.array of shape [bs, K], where likelihoods[i,g] = log P_g(yi|xi)
        # userids: [uid_1, ..., uid_bs]
        for i in range(len(userids)):
            u = userids[i]
            self.log_likelihoods[u] += log_likelihoods[i]

        if self.num_groups > 1:
            # print("\n")
            # print(likelihoods[:,0])
            # print(likelihoods[:,1])
            self.sumcos += self.vecdiff(log_likelihoods[:,0], log_likelihoods[:,1])
            self.Z_sumcos += 1

    def mean_sumcos(self):
        if self.num_groups == 1:
            return 0
        else:
            return self.sumcos/self.Z_sumcos

    def generate_umask(self, userids):
        umask = torch.sparse.LongTensor([len(userids), self.num_users])  # (bs, U)
        for i in range(len(userids)):
            umask[i, userids[i]] = 1
        return umask

    # generate dictionary of userid -> [positions in "userids" list that have userid]
    def generate_uix(self, userids):
        uix = defaultdict(list)
        for ix, val in enumerate(userids):
            uix[val].append(ix)
        return uix

    def group_loss(self, losses, userids):
        # losses = [ (bs), ..., (bs) ]   -> produce e.g. with NLLLoss(reduce=False)
        # losses[g][i] = - log P(y_i | x_i, g)
        stacked_losses = torch.stack(losses)               # (K, bs)
        stacked_losses_t = stacked_losses.transpose(0,1)   # (bs, K)

        # userids = [ uid1, ..., uidn ]
        group_probs = self.gu_posterior(userids)           # (bs, K)

        # calculate expected loss under user-group posterior from previous iteration
        weighted_losses = stacked_losses_t * group_probs   # elementwise multiplication; result is (bs, K)
        total_loss = weighted_losses.sum()                 # sum over groups and instances; result is shape ()

        # accumulate negative log-likelihoods per user
        # self.nll = torch.FloatTensor([self.num_groups, self.num_users]) # (K, U)
        uix = self.generate_uix(userids)
        d_stacked_losses = stacked_losses.detach()
        for userid, ix in uix.items():
            ixt = settings.cd(Variable(LongTensor(ix)))
            nll_for_user = torch.index_select(d_stacked_losses, 1, ixt) # (K, |ix|)
            nll_for_user = nll_for_user.sum(dim=1)                      # (K)
            self.nll[userid,:] += nll_for_user.data

        return total_loss


    def forward(self, userids, *original_inputs):
        predictions = [m(*original_inputs) for m in self.models]         # K x [bs, |Y|]
        prediction_matrix = torch.stack(predictions)                     # [K, bs, |Y|]
        prediction_matrix = torch.transpose(prediction_matrix, 0, 1)     # [bs, K, |Y|]
        prediction_matrix = torch.transpose(prediction_matrix, 1, 2)     # [bs, |Y|, K]

        group_probs = self.gu_posterior(userids)                         # [bs, K]
        group_probs = torch.unsqueeze(group_probs, 2)                    # [bs, K, 1]

        weighted_predictions = torch.bmm(prediction_matrix, group_probs) # [bs, |Y|, 1]
        weighted_predictions = torch.squeeze(weighted_predictions, 2)    # [bs, |Y|]

        return weighted_predictions, prediction_matrix

    def predict(self, group_probs, gold_target, *original_inputs):
        # group_probs: numpy array [K]
        # original_inputs: with bs=1

        # compute output probs based on given group probs
        predictions = [m(*original_inputs) for m in self.models]  # K x [bs, |Y|]     ## this line is the performance bottleneck
        prediction_matrix = torch.stack(predictions)  # [K, bs, |Y|]
        prediction_matrix = torch.transpose(prediction_matrix, 0, 1)  # [bs, K, |Y|]
        prediction_matrix = torch.transpose(prediction_matrix, 1, 2)  # [bs, |Y|, K]

        v_group_probs = Variable(torch.from_numpy(group_probs.reshape(1,-1,1)))           # [1, K, 1]

        weighted_predictions = torch.bmm(prediction_matrix, v_group_probs) # [bs, |Y|, 1]
        weighted_predictions = weighted_predictions.view(-1)             # [|Y|]

        # Bayesian update of group probs
        updated_group_probs = prediction_matrix.data[0, gold_target, :].numpy() * group_probs
        updated_group_probs /= sum(updated_group_probs)

        return weighted_predictions, updated_group_probs



    def get_name(self):
        return self.name


