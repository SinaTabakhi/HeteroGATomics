from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss, ModuleList
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from gat.base_models import HeteroGNN, VCDN
from gat.graph_data_loader import HeteroDataset
from utils import calculate_performance_metrics


class ModelTrainer(pl.LightningModule):
    def __init__(
        self,
        dataset: HeteroDataset,
        num_modalities: int,
        num_classes: int,
        unimodal_model: List[HeteroGNN],
        loss_fn: CrossEntropyLoss,
        multimodal_decoder: Optional[VCDN] = None,
        train_multimodal_decoder: bool = True,
        gat_lr: float = 1e-3,
        gat_wd: float = 1e-3,
        vcdn_lr: float = 5e-2,
        vcdn_wd: float = 1e-3,
        tune_hyperparameters: float = False
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.unimodal_model = ModuleList(unimodal_model)
        self.multimodal_decoder = multimodal_decoder
        self.train_multimodal_decoder = train_multimodal_decoder
        self.loss_fn = loss_fn
        self.gat_lr = gat_lr
        self.gat_wd = gat_wd
        self.vcdn_lr = vcdn_lr
        self.vcdn_wd = vcdn_wd
        self.tune_hyperparameters = tune_hyperparameters

        self.log_metrics = {}

        # activate manual optimization
        self.automatic_optimization = False

    def get_log_metrics(self):
        return self.log_metrics

    def configure_optimizers(self):
        optimizers = []
        lr_schedulers = []

        for modality in range(self.num_modalities):
            optimizer = torch.optim.Adam(list(self.unimodal_model[modality].parameters()),
                                         lr=self.gat_lr,
                                         weight_decay=self.gat_wd)

            scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
            optimizers.append(optimizer)
            lr_schedulers.append(scheduler)

        if self.multimodal_decoder is not None:
            optimizer = torch.optim.Adam(list(self.multimodal_decoder.parameters()),
                                         lr=self.vcdn_lr,
                                         weight_decay=self.vcdn_wd)

            scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
            optimizers.append(optimizer)
            lr_schedulers.append(scheduler)

        return optimizers, lr_schedulers

    def forward(self, data: HeteroDataset, multimodal: bool = False) -> Union[Tensor, List[Tensor]]:
        output = []

        for modality in range(self.num_modalities):
            output.append(
                self.unimodal_model[modality](data[modality].x_dict, data[modality].edge_index_dict,
                                              data[modality].edge_attr_dict)
            )

        if not multimodal:
            return output

        if self.multimodal_decoder is not None:
            return self.multimodal_decoder(output)

        raise TypeError("multimodal_decoder must be defined for multiomics datasets.")

    def training_step(self, train_batch, batch_idx: int):
        is_single_modality = self.num_modalities == 1
        optimizers = self.optimizers()
        lr_schedulers = self.lr_schedulers()
        outputs = self.forward(train_batch, multimodal=False)

        for modality in range(self.num_modalities):
            (optimizers if is_single_modality else optimizers[modality]).zero_grad()
            mask = train_batch[modality]['patient'].train_mask
            loss = self.loss_fn(outputs[modality][mask], train_batch[modality]['patient'].y[mask])
            self.log_metrics.setdefault(f"train_modality_loss_{modality + 1}", []).append(loss.detach().item())
            self.manual_backward(loss)
            (optimizers if is_single_modality else optimizers[modality]).step()
            (lr_schedulers if is_single_modality else lr_schedulers[modality]).step()

        if self.train_multimodal_decoder and self.multimodal_decoder is not None:
            optimizers[-1].zero_grad()
            mask = train_batch[0]['patient'].train_mask
            output = self.forward(train_batch, multimodal=True)
            multi_loss = self.loss_fn(output[mask], train_batch[0]['patient'].y[mask])
            self.log_metrics.setdefault(f"train_multi_loss", []).append(multi_loss.detach().item())
            self.manual_backward(multi_loss)
            optimizers[-1].step()
            lr_schedulers[-1].step()

    def validation_step(self, validation_batch, batch_idx: int):
        if self.tune_hyperparameters:
            if self.multimodal_decoder is not None:
                output = self.forward(validation_batch, multimodal=True)
            else:
                output = self.forward(validation_batch, multimodal=False)[0]

            mask = validation_batch[0]['patient'].val_mask
            pred_val_data = output[mask]
            final_output = F.softmax(pred_val_data, dim=1).detach().cpu().numpy()
            actual_output = validation_batch[0]['patient'].y[mask].detach().cpu()

            if self.num_classes == 2:
                auc = roc_auc_score(actual_output, final_output[:, 1])
                acc = accuracy_score(actual_output, final_output.argmax(1))
                sensitivity, specificity, ppv, npv = calculate_performance_metrics(actual_output, final_output.argmax(1))

                self.log_metrics.setdefault("val_AUROC", []).append(auc)
                self.log_metrics.setdefault("val_Accuracy", []).append(acc)
                self.log_metrics.setdefault("val_NPV", []).append(npv)
                self.log_metrics.setdefault("val_PPV", []).append(ppv)
                self.log_metrics.setdefault("val_Sensitivity", []).append(sensitivity)
                self.log_metrics.setdefault("val_Specificity", []).append(specificity)
            else:
                acc = accuracy_score(actual_output, final_output.argmax(1))
                f1_macro = f1_score(actual_output, final_output.argmax(1), average="macro")
                f1_micro = f1_score(actual_output, final_output.argmax(1), average="micro")
                f1_weighted = f1_score(actual_output, final_output.argmax(1), average="weighted")
                precision = precision_score(actual_output, final_output.argmax(1), average="weighted")
                recall = recall_score(actual_output, final_output.argmax(1), average="weighted")

                self.log_metrics.setdefault("val_Accuracy", []).append(acc)
                self.log_metrics.setdefault("val_F1_macro", []).append(f1_macro)
                self.log_metrics.setdefault("val_F1_micro", []).append(f1_micro)
                self.log_metrics.setdefault("val_F1_weighted", []).append(f1_weighted)
                self.log_metrics.setdefault("val_Precision", []).append(precision)
                self.log_metrics.setdefault("val_Recall", []).append(recall)

    def test_step(self, test_batch, batch_idx: int):
        if self.multimodal_decoder is not None:
            output = self.forward(test_batch, multimodal=True)
        else:
            output = self.forward(test_batch, multimodal=False)[0]

        mask = test_batch[0]['patient'].test_mask
        pred_test_data = output[mask]
        final_output = F.softmax(pred_test_data, dim=1).detach().cpu().numpy()
        actual_output = test_batch[0]['patient'].y[mask].detach().cpu()

        if self.num_classes == 2:
            auc = roc_auc_score(actual_output, final_output[:, 1])
            acc = accuracy_score(actual_output, final_output.argmax(1))
            sensitivity, specificity, ppv, npv = calculate_performance_metrics(actual_output, final_output.argmax(1))

            self.log_metrics.setdefault("test_AUROC", []).append(auc)
            self.log_metrics.setdefault("test_Accuracy", []).append(acc)
            self.log_metrics.setdefault("test_NPV", []).append(npv)
            self.log_metrics.setdefault("test_PPV", []).append(ppv)
            self.log_metrics.setdefault("test_Sensitivity", []).append(sensitivity)
            self.log_metrics.setdefault("test_Specificity", []).append(specificity)
        else:
            acc = accuracy_score(actual_output, final_output.argmax(1))
            f1_macro = f1_score(actual_output, final_output.argmax(1), average="macro")
            f1_micro = f1_score(actual_output, final_output.argmax(1), average="micro")
            f1_weighted = f1_score(actual_output, final_output.argmax(1), average="weighted")
            precision = precision_score(actual_output, final_output.argmax(1), average="weighted")
            recall = recall_score(actual_output, final_output.argmax(1), average="weighted")

            self.log_metrics.setdefault("test_Accuracy", []).append(acc)
            self.log_metrics.setdefault("test_F1_macro", []).append(f1_macro)
            self.log_metrics.setdefault("test_F1_micro", []).append(f1_micro)
            self.log_metrics.setdefault("test_F1_weighted", []).append(f1_weighted)
            self.log_metrics.setdefault("test_Precision", []).append(precision)
            self.log_metrics.setdefault("test_Recall", []).append(recall)

    def _custom_data_loader(self):
        return self.dataset

    def train_dataloader(self):
        return self._custom_data_loader()

    def val_dataloader(self):
        return self._custom_data_loader()

    def test_dataloader(self):
        return self._custom_data_loader()

    def __str__(self) -> str:
        r"""Returns a string representation of the multiomics trainer object.

        Returns:
            str: The string representation of the multiomics trainer object.
        """
        model_str = ["\nModel info:\n", "   Unimodal model:\n"]

        for modality in range(self.num_modalities):
            model_str.append(f"    ({modality + 1}) {self.unimodal_model[modality]}")

        if self.multimodal_decoder is not None:
            model_str.append("\n\n  Multimodal decoder:\n")
            model_str.append(f"    {self.multimodal_decoder}")

        return "".join(model_str)
