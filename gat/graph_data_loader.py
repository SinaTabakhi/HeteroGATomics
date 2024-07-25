import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split


class HeteroDataset:
    def __init__(self, raw_data, sparsity_rate: float, tune_hyperparameters: bool = False, seed: int = 24):
        self.raw_data = raw_data
        self.sparsity_rate = sparsity_rate
        self.tune_hyperparameters = tune_hyperparameters
        self.seed = seed
        self.hetero_datasets = []

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index):
        return self.hetero_datasets

    def get(self, omics_idx):
        return self.hetero_datasets[omics_idx]

    def create_hetero_data(self, feat_removal_idx: int = None, omics_removal_idx: int = None):
        """ create heterogeneous data

        Args:
            feat_removal_idx (int, optional): The index of the feature to be excluded for biomarker identification.
            omics_removal_idx (int, optional): The index of the omics modality from which the feature will be removed
                for biomarker identification.
        """
        for omics_idx in range(self.raw_data.num_omics):
            if omics_idx == omics_removal_idx:
                self._create_hetero_data(omics_idx, feat_removal_idx)
            else:
                self._create_hetero_data(omics_idx)

    def _create_hetero_data(self, omics_idx: int, feat_removal_idx: int = None):
        hetero_data = HeteroData()
        single_data = self.raw_data.get(omics_idx)
        sparse_adj_mat, sparse_feat_corr, edge_pheromone, node_relevance, node_pheromone, selected_subset = single_data.get_data_structure()

        """
         Build patient similarity network:
             - Combine train and test datasets and combine train and test labels
             - Store the train and test sample indices in train_mask and test_mask
         """
        data = pd.concat([single_data.train_data, single_data.test_data], axis=0)

        # use for biomarker identification
        if feat_removal_idx is not None:
            data.iloc[:, feat_removal_idx] = 0

        label = pd.concat([single_data.train_label, single_data.test_label], axis=0)

        hetero_data['patient'].x = torch.tensor(data.values, dtype=torch.float)
        hetero_data['patient'].y = torch.tensor(label.to_numpy().flatten())
        hetero_data['patient'].test_mask = torch.tensor(np.arange(single_data.num_samples) >= single_data.num_train_samples)

        if self.tune_hyperparameters:
            train_index, val_index = train_test_split(range(single_data.num_train_samples), test_size=0.1,
                                                      random_state=self.seed, stratify=single_data.train_label)
            train_index.sort()
            val_index.sort()

            train_mask = np.zeros(single_data.num_samples, dtype=bool)
            train_mask[train_index] = True
            train_mask = torch.tensor(train_mask, dtype=torch.bool)

            val_mask = np.zeros(single_data.num_samples, dtype=bool)
            val_mask[val_index] = True
            val_mask = torch.tensor(val_mask, dtype=torch.bool)

            hetero_data['patient'].train_mask = train_mask
            hetero_data['patient'].val_mask = val_mask
        else:
            hetero_data['patient'].train_mask = torch.tensor(np.arange(single_data.num_samples) < single_data.num_train_samples)

        patient_adj_mat = self._build_patient_sim_net(data)
        hetero_data['patient', 'similar', 'patient'].edge_index = torch.tensor(np.concatenate(patient_adj_mat.nonzero()).flatten().reshape(2, -1), dtype=torch.long)
        hetero_data['patient', 'similar', 'patient'].edge_attr = torch.tensor(patient_adj_mat.data.reshape(-1, 1), dtype=torch.float)  # [num_edges_similar, 1]

        """
        Build feature similarity network:
            - Construct edge_index by adjacency matrix (sparse_adj_mat).
            - Construct edge_attr by the combination of correlation (feat_corr) and pheromone values (edge_pheromone).
                Only edges that exist in the adjacency matrix are considered.
        """
        dense_adj_mat = sparse_adj_mat.todense()
        feat_corr = sparse_feat_corr.todense()
        feat_corr = feat_corr + feat_corr.transpose()  # sparse_feat_corr retains only the upper triangle of the correlation matrix
        feat_corr[~dense_adj_mat] = 0.0
        edge_pheromone[~dense_adj_mat] = 0.0

        # Retain only the rows and columns present in the selected feature subset
        dense_adj_mat = dense_adj_mat[selected_subset][:, selected_subset]
        sparse_adj_mat = sp.csr_matrix(dense_adj_mat)
        feature_edge_index = np.concatenate(sparse_adj_mat.nonzero()).flatten().reshape(2, -1)

        feat_corr = feat_corr[selected_subset][:, selected_subset]
        feature_edge_attr_corr = feat_corr[sparse_adj_mat.nonzero()].reshape(-1, 1)

        edge_pheromone = edge_pheromone[selected_subset][:, selected_subset]
        feature_edge_attr_pheromone = edge_pheromone[sparse_adj_mat.nonzero()].reshape(-1, 1)

        feature_edge_attr = np.hstack((feature_edge_attr_corr, feature_edge_attr_pheromone))

        # Add extracted feature info (node relevance and node pheromone) to the feature similarity network
        node_relevance = node_relevance.iloc[selected_subset]
        node_pheromone = node_pheromone[selected_subset]
        extracted_feat_info = np.column_stack((node_relevance.values, node_pheromone)).reshape(-1, 2)

        # Use for biomarker identification
        if feat_removal_idx is not None:
            extracted_feat_info[feat_removal_idx, :] = 0

        extracted_feat_info = torch.tensor(extracted_feat_info, dtype=torch.float)

        feat_x = torch.tensor(data.transpose().values, dtype=torch.float)
        feat_x_new = torch.cat((feat_x, extracted_feat_info), dim=1)

        hetero_data['feature'].x = feat_x_new
        hetero_data['feature'].feat_name = data.columns.tolist()
        hetero_data['feature', 'similar', 'feature'].edge_index = torch.tensor(feature_edge_index, dtype=torch.long)
        hetero_data['feature', 'similar', 'feature'].edge_attr = torch.tensor(feature_edge_attr, dtype=torch.float)

        """
        Build feature and patient relations
        """
        belong_from = np.repeat(np.arange(len(selected_subset)), single_data.num_samples)
        belong_to = np.tile(np.arange(single_data.num_samples), len(selected_subset))
        belong = np.vstack((belong_from, belong_to))

        hetero_data['feature', 'belong', 'patient'].edge_index = torch.tensor(belong, dtype=torch.long)

        self.hetero_datasets.insert(omics_idx, hetero_data)

    def _find_similarity_threshold(self, sim_net, ascending=False):
        sorted_correlation = sim_net.unstack().sort_values(ascending=ascending)
        threshold_index = int(sim_net.size * self.sparsity_rate)
        threshold_value = sorted_correlation.iloc[threshold_index]

        return threshold_value

    def _build_patient_sim_net(self, data):
        patient_adj_mat = data.transpose().corr(method='pearson')
        patient_adj_mat = patient_adj_mat.map(abs)

        similarity_threshold = self._find_similarity_threshold(patient_adj_mat, ascending=True)
        non_zero_entries = patient_adj_mat >= similarity_threshold
        patient_adj_mat = patient_adj_mat * non_zero_entries
        patient_adj_mat = sp.csr_matrix(patient_adj_mat)

        return patient_adj_mat

    def get_feature_name(self, omics_idx, feat_idx):
        return self.hetero_datasets[omics_idx]['feature'].feat_name[feat_idx]