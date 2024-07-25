import os
import time
import argparse
import warnings
import pickle
import gzip
import copy
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl

from gat.graph_data_loader import HeteroDataset
from gat.architecture import NewModel
from utils import seed_everything, create_file, save_output, sort_file, select_top_feats, is_directory_empty, load_dataset_indices
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

    if is_directory_empty(cfg.RESULT.SAVE_RICH_DATA_DIR):
        raise Exception("Perform feature selection first")

    # ---- setup folders and paths ----
    if not os.path.exists(cfg.RESULT.OUTPUT_DIR) and cfg.RESULT.SAVE_RESULT:
        os.makedirs(cfg.RESULT.OUTPUT_DIR)
    if not os.path.exists(cfg.RESULT.SAVE_MODEL_DIR) and cfg.RESULT.SAVE_MODEL:
        os.makedirs(cfg.RESULT.SAVE_MODEL_DIR)

    main_folder = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME)
    raw_file_paths = [(os.path.join(main_folder, f"{omics}.csv"), omics) for omics in cfg.DATASET.OMICS]
    raw_label_path = os.path.join(main_folder, f"ClinicalMatrix.csv")

    if cfg.RESULT.SAVE_RESULT:
        output_gat_file = os.path.join(cfg.RESULT.OUTPUT_DIR, f'HeteroGATomics_gat_{cfg.DATASET.NAME}.csv')
        sorted_output_gat_file = os.path.join(cfg.RESULT.OUTPUT_DIR, f'HeteroGATomics_gat_{cfg.DATASET.NAME}_sorted.csv')
        output_gat_file_time = os.path.join(cfg.RESULT.OUTPUT_DIR, f'HeteroGATomics_gat_{cfg.DATASET.NAME}_time.csv')
        sorted_output_gat_file_time = os.path.join(cfg.RESULT.OUTPUT_DIR, f'HeteroGATomics_gat_{cfg.DATASET.NAME}_time_sorted.csv')

        if cfg.SOLVER.TUNE_HYPER:
            output_hyperparameter = os.path.join(cfg.RESULT.OUTPUT_DIR, f'HeteroGATomics_hyper_{cfg.DATASET.NAME}.csv')

        create_file(file_dir=output_gat_file, header=cfg.RESULT.FILE_HEADER_GAT)
        create_file(file_dir=output_gat_file_time, header=cfg.RESULT.FILE_HEADER_GAT_TIME)

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

    final_gat_results = {}
    gat_results = defaultdict(lambda: defaultdict(list))
    time_results = defaultdict(list)

    for fold_idx in range(cfg.DATASET.NUM_FOLDS):
        print(f"==> Loading data from fold {fold_idx + 1}...")
        fold_multiomics = prepare_data(main_folder, fold_idx, multiomics, cfg)
        node_pheromones = fold_multiomics.get_node_pheromone()
        node_relevances = fold_multiomics.get_node_relevance()

        for feat_size in cfg.GAT.FINAL_FEAT_SIZES:
            final_feat_subset = select_top_feats(node_pheromones, node_relevances, feat_size, fold_multiomics.num_omics,
                                                 cfg.ACO.SELECTION_RATE)

            multiomics_deepcopy = copy.deepcopy(fold_multiomics)
            multiomics_deepcopy.reduce_dimensionality(final_feat_subset, feat_size)

            start_time = time.time()
            hetero_data = HeteroDataset(multiomics_deepcopy, cfg.DATASET.PATIENT_SPARSITY_RATES,
                                        tune_hyperparameters=cfg.SOLVER.TUNE_HYPER,
                                        seed=cfg.SOLVER.SEED)
            hetero_data.create_hetero_data()

            # ---- setup model ----
            print("\n   ==> Building model...")
            new_model = NewModel(dataset=hetero_data,
                                 num_modalities=multiomics_deepcopy.num_omics,
                                 num_classes=multiomics_deepcopy.num_classes,
                                 gat_num_layers=cfg.GAT.NUM_LAYERS,
                                 gat_num_heads=cfg.GAT.NUM_HEADS,
                                 gat_hidden_dim=cfg.GAT.HIDDEN_DIM,
                                 gat_dropout_rate=cfg.GAT.DROPOUT_RATE,
                                 gat_lr_pretrain=cfg.GAT.LR_PRETRAIN,
                                 gat_lr=cfg.GAT.LR,
                                 gat_wd=cfg.GAT.WD,
                                 vcdn_lr=cfg.VCDN.LR,
                                 vcdn_wd=cfg.VCDN.WD,
                                 tune_hyperparameters=cfg.SOLVER.TUNE_HYPER
                                 )

            # ---- setup pretraining model and trainer ----
            print("\n   ==> Pretrain model...")
            model = new_model.get_model(pretrain=True)
            trainer_pretrain = pl.Trainer(
                max_epochs=cfg.SOLVER.MAX_EPOCHS_PRETRAIN,
                default_root_dir=cfg.RESULT.LIGHTNING_LOG_DIR,
                accelerator="auto",
                devices="auto",
                enable_model_summary=False,
                log_every_n_steps=1
            )
            trainer_pretrain.fit(model)

            # ---- setup training model and trainer ----
            print("\n   ==> Training model...")
            model = new_model.get_model(pretrain=False)
            trainer = pl.Trainer(
                max_epochs=cfg.SOLVER.MAX_EPOCHS,
                default_root_dir=cfg.RESULT.LIGHTNING_LOG_DIR,
                accelerator="auto",
                devices="auto",
                enable_model_summary=False,
                log_every_n_steps=1
            )
            trainer.fit(model)

            if cfg.RESULT.SAVE_MODEL:
                save_model_name = cfg.RESULT.SAVE_MODEL_TMPL.format(dataset_name=cfg.DATASET.NAME, fold_idx=fold_idx + 1,
                                                                    feat_size=feat_size)
                trainer.save_checkpoint(os.path.join(cfg.RESULT.SAVE_MODEL_DIR, save_model_name))

            # ---- test model ----
            print("\n   ==> Testing model...")
            trainer.test(model)
            end_time = time.time()
            running_time = end_time - start_time

            time_results[feat_size].append(running_time)
            if cfg.RESULT.SAVE_RESULT:
                time_result = [feat_size, fold_idx + 1, running_time]
                save_output(output_gat_file_time, time_result)

            final_gat_results.setdefault(feat_size, []).append(model.get_log_metrics())

            for metric_key, metric_value in model.get_log_metrics().items():
                if metric_key.startswith("test_"):
                    gat_results[feat_size][metric_key.replace("test_", "")].append(metric_value[0])
                    if cfg.RESULT.SAVE_RESULT:
                        result = [feat_size, metric_key.replace("test_", ""), fold_idx + 1, metric_value[0]]
                        save_output(output_gat_file, result)

    if cfg.RESULT.SAVE_RESULT:
        sort_file(output_gat_file, sorted_output_gat_file, by=cfg.RESULT.FILE_HEADER_GAT[0:3])
        sort_file(output_gat_file_time, sorted_output_gat_file_time, by=cfg.RESULT.FILE_HEADER_GAT_TIME[0:2])

    print(f"\n==> Showing results...")
    for feat_size, metrics in gat_results.items():
        exe_time = round(np.mean(time_results[feat_size]))
        print(f"Feature size {feat_size} (execution time: {exe_time} seconds)")
        for metric_name, values in metrics.items():
            average = np.mean(values)
            std = np.std(values)
            print(f"    - {metric_name}: {average:.3f}±{std:.3f}")

    if cfg.SOLVER.TUNE_HYPER:
        print(f"\n==> Saving hyperparameter results...")
        average_gat_results = {}
        for feat_size, results in final_gat_results.items():
            sum_metrics = defaultdict(lambda: np.zeros(cfg.SOLVER.MAX_EPOCHS + 1))
            count_folds = len(results)

            # Sum up the metrics for each fold
            for result in results:
                for metric, values in result.items():
                    if metric.startswith("val_"):
                        sum_metrics[metric.replace("val_", "")] += np.array(values)

            # Calculate the average over all folds for each epoch
            average_metrics = {metric: values / count_folds for metric, values in sum_metrics.items()}
            average_gat_results[feat_size] = average_metrics

        output_csv_rows = []
        for feat_size, metrics in average_gat_results.items():
            output_csv_rows.append([f"Feature size: {feat_size}"])
            for metric, average in metrics.items():
                row = [metric] + average.tolist()
                output_csv_rows.append(row)
            output_csv_rows.append([])

        save_output(output_hyperparameter, output_csv_rows, multi_rows=True)


if __name__ == '__main__':
    main()
