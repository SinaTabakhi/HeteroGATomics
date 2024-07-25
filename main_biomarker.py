import os
import argparse
import warnings
import pickle
import gzip
import copy

import pytorch_lightning as pl
import pandas as pd

from gat.graph_data_loader import HeteroDataset
from gat.architecture import NewModel
from gat.trainer import ModelTrainer
from utils import seed_everything, select_top_feats, is_directory_empty, load_dataset_indices
from configs import get_cfg_defaults
from raw_data_loader import MultiOmicsDataset


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="HeteroGATomics for multiomics data integration")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    args = parser.parse_args()

    return args


def prepare_data(main_folder, fold_idx, multiomics, cfg):
    fold_dir = os.path.join(main_folder, f"{fold_idx + 1}")
    train_index, test_index = load_dataset_indices(fold_dir)
    multiomics_copy = copy.deepcopy(multiomics)

    train_index.sort()
    test_index.sort()
    multiomics_copy.set_train_test(train_index, test_index)
    multiomics_copy.config_components()

    load_data_name = cfg.RESULT.SAVE_RICH_DATA_TMPL.format(dataset_name=cfg.DATASET.NAME, fold_idx=fold_idx + 1)

    with gzip.open(os.path.join(cfg.RESULT.SAVE_RICH_DATA_DIR, load_data_name), 'rb') as file:
        loaded_data = pickle.load(file)
        loaded_node_pheromones, loaded_edge_pheromones = loaded_data
        multiomics_copy.set_data_structure(loaded_node_pheromones, loaded_edge_pheromones)

    return multiomics_copy


def main():
    warnings.filterwarnings(action="ignore")

    # ---- setup configs ----
    args = arg_parse()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    seed_everything(cfg.SOLVER.SEED, workers=True)

    # ---- setup folders and paths ----
    if not os.path.exists(cfg.RESULT.OUTPUT_DIR) and cfg.RESULT.SAVE_RESULT:
        os.makedirs(cfg.RESULT.OUTPUT_DIR)
    if is_directory_empty(cfg.RESULT.SAVE_MODEL_DIR):
        raise Exception("Perform GAT prediction first")

    main_folder = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME)
    raw_file_paths = [(os.path.join(main_folder, f"{omics}.csv"), omics) for omics in cfg.DATASET.OMICS]
    raw_label_path = os.path.join(main_folder, f"ClinicalMatrix.csv")

    # ---- setup multiomics dataset ----
    multiomics = MultiOmicsDataset(
        dataset_name=cfg.DATASET.NAME,
        raw_file_paths=raw_file_paths,
        raw_label_path=raw_label_path,
        num_omics=len(cfg.DATASET.OMICS),
        num_classes=cfg.DATASET.NUM_CLASSES,
        init_pheromone_val=cfg.ACO.INIT_PHEROMONE,
        sparsity_rates=cfg.DATASET.FEATURE_SPARSITY_RATES
    )
    print(multiomics)

    feat_freq_across_folds = {}

    for fold_idx in range(cfg.DATASET.NUM_FOLDS):
        feat_imp = []
        print(f"==> Loading data from fold {fold_idx + 1}...")
        fold_multiomics = prepare_data(main_folder, fold_idx, multiomics, cfg)
        node_pheromones = fold_multiomics.get_node_pheromone()
        node_relevances = fold_multiomics.get_node_relevance()

        final_feat_subset = select_top_feats(node_pheromones, node_relevances, cfg.BIOMK.FEAT_SIZE, fold_multiomics.num_omics,
                                             cfg.ACO.SELECTION_RATE)

        fold_multiomics.reduce_dimensionality(final_feat_subset, cfg.BIOMK.FEAT_SIZE)

        hetero_data = HeteroDataset(fold_multiomics, cfg.DATASET.PATIENT_SPARSITY_RATES)
        hetero_data.create_hetero_data()

        # ---- setup model ----
        print("\n   ==> Building model...")
        new_model = NewModel(dataset=hetero_data,
                             num_modalities=fold_multiomics.num_omics,
                             num_classes=fold_multiomics.num_classes,
                             gat_num_layers=cfg.GAT.NUM_LAYERS,
                             gat_num_heads=cfg.GAT.NUM_HEADS,
                             gat_hidden_dim=cfg.GAT.HIDDEN_DIM,
                             gat_dropout_rate=cfg.GAT.DROPOUT_RATE,
                             gat_lr_pretrain=cfg.GAT.LR_PRETRAIN,
                             gat_lr=cfg.GAT.LR,
                             gat_wd=cfg.GAT.WD,
                             vcdn_lr=cfg.VCDN.LR,
                             vcdn_wd=cfg.VCDN.WD
                             )

        model = new_model.get_model(pretrain=False)
        load_model_name = cfg.RESULT.SAVE_MODEL_TMPL.format(dataset_name=cfg.DATASET.NAME,
                                                            fold_idx=fold_idx + 1,
                                                            feat_size=cfg.BIOMK.FEAT_SIZE)

        model = ModelTrainer.load_from_checkpoint(checkpoint_path=os.path.join(cfg.RESULT.SAVE_MODEL_DIR, load_model_name),
                                                  dataset=model.dataset,
                                                  num_modalities=model.num_modalities,
                                                  num_classes=model.num_classes,
                                                  unimodal_model=model.unimodal_model,
                                                  loss_fn=model.loss_fn,
                                                  multimodal_decoder=model.multimodal_decoder,
                                                  train_multimodal_decoder=model.train_multimodal_decoder,
                                                  tune_hyperparameters=model.tune_hyperparameters)

        trainer = pl.Trainer(
            max_epochs=cfg.SOLVER.MAX_EPOCHS,
            default_root_dir=cfg.RESULT.LIGHTNING_LOG_DIR,
            accelerator="auto",
            devices="auto",
            enable_model_summary=False,
            log_every_n_steps=1
        )

        # ---- test model ----
        print("\n   ==> Testing model...")
        trainer.test(model)

        if fold_multiomics.num_classes == 2:
            original_total_value = model.get_log_metrics()["test_AUROC"][0]
        else:
            original_total_value = model.get_log_metrics()["test_F1_weighted"][0]

        for omics_idx in range(fold_multiomics.num_omics):
            for feat_idx in range(fold_multiomics.get(omics_idx).train_data.shape[1]):
                hetero_data = HeteroDataset(fold_multiomics, cfg.DATASET.PATIENT_SPARSITY_RATES)
                hetero_data.create_hetero_data(feat_removal_idx=feat_idx, omics_removal_idx=omics_idx)
                feat_name = hetero_data.get_feature_name(omics_idx, feat_idx)

                # ---- setup model ----
                print("\n==> Building model...")
                new_model = NewModel(dataset=hetero_data,
                                     num_modalities=fold_multiomics.num_omics,
                                     num_classes=fold_multiomics.num_classes,
                                     gat_num_layers=cfg.GAT.NUM_LAYERS,
                                     gat_num_heads=cfg.GAT.NUM_HEADS,
                                     gat_hidden_dim=cfg.GAT.HIDDEN_DIM,
                                     gat_dropout_rate=cfg.GAT.DROPOUT_RATE,
                                     gat_lr_pretrain=cfg.GAT.LR_PRETRAIN,
                                     gat_lr=cfg.GAT.LR,
                                     gat_wd=cfg.GAT.WD,
                                     vcdn_lr=cfg.VCDN.LR,
                                     vcdn_wd=cfg.VCDN.WD
                                     )
                model = new_model.get_model(pretrain=False)
                load_model_name = cfg.RESULT.SAVE_MODEL_TMPL.format(dataset_name=cfg.DATASET.NAME,
                                                                    fold_idx=fold_idx + 1,
                                                                    feat_size=cfg.BIOMK.FEAT_SIZE)

                model = ModelTrainer.load_from_checkpoint(checkpoint_path=os.path.join(cfg.RESULT.SAVE_MODEL_DIR, load_model_name),
                                                          dataset=model.dataset,
                                                          num_modalities=model.num_modalities,
                                                          num_classes=model.num_classes,
                                                          unimodal_model=model.unimodal_model,
                                                          loss_fn=model.loss_fn,
                                                          multimodal_decoder=model.multimodal_decoder,
                                                          train_multimodal_decoder=model.train_multimodal_decoder,
                                                          tune_hyperparameters=model.tune_hyperparameters)

                trainer = pl.Trainer(
                    max_epochs=cfg.SOLVER.MAX_EPOCHS,
                    default_root_dir=cfg.RESULT.LIGHTNING_LOG_DIR,
                    accelerator="auto",
                    devices="auto",
                    enable_model_summary=False,
                    log_every_n_steps=1
                )

                # ---- test model ----
                print("\n==> Testing model...")
                trainer.test(model)

                feat_identifier = (feat_name, omics_idx)
                if fold_multiomics.num_classes == 2:
                    feat_imp.append((feat_identifier, original_total_value - model.get_log_metrics()["test_AUROC"][0]))
                else:
                    feat_imp.append((feat_identifier, original_total_value - model.get_log_metrics()["test_F1_weighted"][0]))

        # Update the value sum and frequency of top features across folds
        for (feat_name, omics_idx), score in feat_imp:
            feat_identifier = (feat_name, omics_idx)
            if feat_identifier in feat_freq_across_folds:
                current_score, freq = feat_freq_across_folds[feat_identifier]
                feat_freq_across_folds[feat_identifier] = (current_score + score, freq + 1)
            else:
                feat_freq_across_folds[feat_identifier] = (score, 1)

    # Normalize frequencies and values by dividing by the number of folds
    for feat_identifier, (score_sum, freq) in feat_freq_across_folds.items():
        normalized_score = score_sum / freq
        feat_freq_across_folds[feat_identifier] = (normalized_score, freq)

    # First, sort by frequencies in descending order to ensure the highest frequencies come first
    pre_sorted_feat_info = sorted(feat_freq_across_folds.items(), key=lambda item: item[1][1], reverse=True)

    # Then, sort by values in descending order, keeping the highest frequencies first when values are equal
    sorted_feat_info = sorted(pre_sorted_feat_info, key=lambda item: item[1][0], reverse=True)[:cfg.BIOMK.NUM_TOP_BIOMARKERS]

    df_feat_info = []
    for rank, ((feat_name, omics_idx), (total_score, freq)) in enumerate(sorted_feat_info, start=1):
        omics_name = cfg.DATASET.OMICS[omics_idx]
        data_row = {
            "Rank": rank,
            "Biomarker ID": feat_name,
            "Omic": omics_name,
            "Score": round(total_score, 5)
        }
        df_feat_info.append(data_row)

    df = pd.DataFrame(df_feat_info)
    df.set_index("Rank", inplace=True)
    print(df)


if __name__ == '__main__':
    main()
