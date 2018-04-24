import torch

import settings
from models import PureGRU, EmbeddingGRU


class GroupModel:
    def __init__(self, num_groups, modelclass, **kwargs):
        self.num_groups = num_groups
        self.models = [modelclass(**kwargs) for i in range(num_groups)]

        # TODO - constant tensor for user->group probs

    def forward(self, x, userids):
        predictions = [m.forward(x) for m in self.models]   # K x [|Y|]
        prediction_matrix = torch.stack(predictions)        # [K, |Y|]

        # TODO - multiply with user->group lookups



gm = GroupModel(2, EmbeddingGRU, hidden_size=5, num_layers=1, embedding_dim=2)
