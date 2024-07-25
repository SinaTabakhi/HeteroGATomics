from raw_data_loader import MultiOmicsDataset
from utils import evaluate_feat_subset


class Agent:
    r"""The definition of an agent within the multi-agent system

    Args:
        num_feats (int): Number of features to be selected by each agent.
    """
    def __init__(
            self,
            num_feats: int,
    ) -> None:
        self.num_feats = num_feats
        # feat_set is a dictionary in which
        # keys show the index of the modality and
        # values indicate the indices of the selected features in a specific modality
        self.feat_set = {}
        # edge_set is a dictionary in which
        # keys show the index of the modality and
        # values indicate a tuple (modality index, feature index) of the selected features in a specific modality
        self.edge_set = {}
        self.last_feat = None
        self.last_modality = None
        self.rel_val = 0
        self.cor_val = 0
        self.fit_val = 0
        self.reset_parameters()

    def reset_parameters(self):
        self.feat_set = {}
        self.edge_set = {}
        self.last_feat = None
        self.last_modality = None
        self.rel_val = 0
        self.cor_val = 0
        self.fit_val = 0

    def add_next_feat(self, feat_idx, modality_idx, rel_val, sum_corr=0):
        # add the selected edge to the edge_set dictionary
        if self.last_modality == modality_idx:
            modality_value = self.edge_set.get(self.last_modality, [])
            if self.last_feat < feat_idx:
                modality_value.append((self.last_feat, feat_idx))
            else:
                modality_value.append((feat_idx, self.last_feat))
            self.edge_set[self.last_modality] = modality_value

        self.last_feat = feat_idx
        self.last_modality = modality_idx

        # add the selected feature to the feat_set dictionary
        modality_value = self.feat_set.get(self.last_modality, [])
        modality_value.append(self.last_feat)
        self.feat_set[self.last_modality] = modality_value

        # update total relevance and correlation values of the agent
        self.rel_val += rel_val
        self.cor_val += sum_corr

    def eval_feat_subset(self, num_corr: int, data: MultiOmicsDataset):
        training_data = data.concatenate_data(self.feat_set, is_train=True)
        acc = evaluate_feat_subset(training_data.values, data.get(0).train_label.values.ravel(), data.num_classes)
        self.rel_val /= self.num_feats
        self.cor_val /= num_corr

        self.fit_val = (self.rel_val + acc + (1.0 - self.cor_val)) / 3.0

    def __str__(self) -> str:
        return f"""
Agent definition
    Selected features:      {self.feat_set}
    Selected edges:         {self.edge_set}
    Last selected feature:  {self.last_feat}     
    Last selected modality: {self.last_modality}
    Total relevance:        {self.rel_val}
    Total correlation:      {self.cor_val}
    Fitness value:          {self.fit_val}
"""
