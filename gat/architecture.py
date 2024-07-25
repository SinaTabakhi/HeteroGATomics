from typing import List, Optional

import torch
from torch.nn import CrossEntropyLoss

from gat.graph_data_loader import HeteroDataset
from gat.trainer import ModelTrainer
from gat.base_models import HeteroGNN, VCDN


class NewModel:
    def __init__(self,
                 dataset: HeteroDataset,
                 num_modalities: int,
                 num_classes: int,
                 gat_num_layers: int,
                 gat_num_heads: int,
                 gat_hidden_dim: List[int],
                 gat_dropout_rate: float,
                 gat_lr_pretrain: float,
                 gat_lr: float,
                 gat_wd: float,
                 vcdn_lr: float,
                 vcdn_wd: float,
                 tune_hyperparameters: bool = False) -> None:
        self.dataset = dataset
        self.unimodal_model: List[HeteroGNN] = []
        self.multimodal_decoder: Optional[VCDN] = None
        self.loss_function = CrossEntropyLoss()
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.gat_num_layers = gat_num_layers
        self.gat_num_heads = gat_num_heads
        self.gat_hidden_dim = gat_hidden_dim
        self.gat_dropout_rate = gat_dropout_rate
        self.gat_lr_pretrain = gat_lr_pretrain
        self.gat_lr = gat_lr
        self.gat_wd = gat_wd
        self.vcdn_lr = vcdn_lr
        self.vcdn_wd = vcdn_wd
        self.vcdn_hidden_dim = pow(self.num_classes, self.num_modalities)
        self.tune_hyperparameters = tune_hyperparameters

        self._create_model()

    def _create_model(self) -> None:
        for modality in range(self.num_modalities):
            self.unimodal_model.append(
                HeteroGNN(
                    hidden_channels=self.gat_hidden_dim,
                    out_channels=self.num_classes,
                    num_layers=self.gat_num_layers,
                    num_heads=self.gat_num_heads,
                    dropout=self.gat_dropout_rate
                )
            )

        # Initialize lazy modules
        with torch.no_grad():
            for modality in range(self.num_modalities):
                self.unimodal_model[modality](self.dataset.get(modality).x_dict,
                                              self.dataset.get(modality).edge_index_dict,
                                              self.dataset.get(modality).edge_attr_dict)

        if self.num_modalities >= 2:
            self.multimodal_decoder = VCDN(
                num_modalities=self.num_modalities, num_classes=self.num_classes, hidden_dim=self.vcdn_hidden_dim
            )

    def get_model(self, pretrain: bool = False) -> ModelTrainer:
        if pretrain:
            multimodal_model = None
            train_multimodal_decoder = False
            gat_lr = self.gat_lr_pretrain
        else:
            multimodal_model = self.multimodal_decoder
            train_multimodal_decoder = True
            gat_lr = self.gat_lr

        model = ModelTrainer(
            dataset=self.dataset,
            num_modalities=self.num_modalities,
            num_classes=self.num_classes,
            unimodal_model=self.unimodal_model,
            multimodal_decoder=multimodal_model,
            train_multimodal_decoder=train_multimodal_decoder,
            loss_fn=self.loss_function,
            gat_lr=gat_lr,
            gat_wd=self.gat_wd,
            vcdn_lr=self.vcdn_lr,
            vcdn_wd=self.vcdn_wd,
            tune_hyperparameters=self.tune_hyperparameters
        )

        return model

    def __str__(self) -> str:
        r"""Returns a string representation of the model object.

        Returns:
            str: The string representation of the model object.
        """
        return self.get_model().__str__()
