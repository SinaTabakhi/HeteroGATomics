import os
import time
import argparse
import warnings
import pickle
import gzip
import copy
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from concurrent.futures import ProcessPoolExecutor as Pool
from xgboost import XGBClassifier

from raw_data_loader import MultiOmicsDataset
from feature_selection.multi_agent_system import MultiAgentSystem
from utils import seed_everything, evaluate_model, select_top_feats, load_dataset_indices, create_file, save_output, sort_file
from configs import get_cfg_defaults


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="HeteroGATomics for multiomics data integration")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    args = parser.parse_args()

    return args


def prepare_data(main_folder, fold_idx, multiomics):
    fold_dir = os.path.join(main_folder, f"{fold_idx + 1}")
    train_index, test_index = load_dataset_indices(fold_dir)
    multiomics_copy = copy.deepcopy(multiomics)

    train_index.sort()
    test_index.sort()
    multiomics_copy.set_train_test(train_index, test_index)
    multiomics_copy.config_components()

    return multiomics_copy


def select_features(args):
    fold_idx, multiomics, agent_max_workers, cfg = args

    start_time = time.time()
    multi_agent = MultiAgentSystem(
        max_workers=agent_max_workers,
        dataset=multiomics,
        max_iters=cfg.ACO.MAX_ITERS,
        num_agents=cfg.ACO.NUM_AGENTS,
        num_feats=cfg.ACO.FIX_FEAT_SIZE,
        q0=cfg.ACO.Q0,
        node_discount_rate=cfg.ACO.NODE_DISC_RATE,
        edge_discount_rate=cfg.ACO.EDGE_DISC_RATE,
        prob_discount_rate=cfg.ACO.PROB_DISC_RATE
    )

    print(f"\n==> Performing feature selection for fold {fold_idx + 1}...")
    print(multi_agent)
    multi_agent.run()

    end_time = time.time()
    running_time = end_time - start_time

    return fold_idx, running_time, multiomics


def main():
    warnings.filterwarnings(action="ignore")

    # ---- setup configs ----
    args = arg_parse()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    seed_everything(cfg.SOLVER.SEED, workers=True)

    # ---- setup CPU cores ----
    agent_max_workers = min(cfg.ACO.NUM_AGENTS, os.cpu_count())
    fold_max_workers = max(1, (os.cpu_count() - 2) // agent_max_workers)

    # ---- setup folders and paths ----
    if not os.path.exists(cfg.RESULT.OUTPUT_DIR) and cfg.RESULT.SAVE_RESULT:
        os.makedirs(cfg.RESULT.OUTPUT_DIR)
    if not os.path.exists(cfg.RESULT.SAVE_RICH_DATA_DIR) and cfg.RESULT.SAVE_RICH_DATA:
        os.makedirs(cfg.RESULT.SAVE_RICH_DATA_DIR)

    main_folder = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME)
    raw_file_paths = [(os.path.join(main_folder, f"{omics}.csv"), omics) for omics in cfg.DATASET.OMICS]
    raw_label_path = os.path.join(main_folder, f"ClinicalMatrix.csv")

    if cfg.RESULT.SAVE_RESULT:
        output_file = os.path.join(cfg.RESULT.OUTPUT_DIR, f'HeteroGATomics_{cfg.DATASET.NAME}.csv')
        sorted_output_file = os.path.join(cfg.RESULT.OUTPUT_DIR, f'HeteroGATomics_{cfg.DATASET.NAME}_sorted.csv')
        output_file_time = os.path.join(cfg.RESULT.OUTPUT_DIR, f'HeteroGATomics_{cfg.DATASET.NAME}_time.csv')
        sorted_output_file_time = os.path.join(cfg.RESULT.OUTPUT_DIR, f'HeteroGATomics_{cfg.DATASET.NAME}_time_sorted.csv')

        create_file(file_dir=output_file, header=cfg.RESULT.FILE_HEADER_CLF)
        create_file(file_dir=output_file_time, header=cfg.RESULT.FILE_HEADER_TIME)

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

    fold_multiomics = []
    for fold_idx in range(cfg.DATASET.NUM_FOLDS):
        print(f"==> Loading data from fold {fold_idx + 1}...")
        fold_multiomics.append(prepare_data(main_folder, fold_idx, multiomics))

    print()
    with Pool(max_workers=fold_max_workers) as pool:
        fold_results = pool.map(select_features, [(fold_idx, fold_multiomics[fold_idx], agent_max_workers, cfg)
                                                  for fold_idx in range(cfg.DATASET.NUM_FOLDS)])

    cls_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    time_results = defaultdict(list)

    print(f"\n==> Collecting information...")
    for fold_idx, running_time, rich_multiomics in fold_results:
        node_pheromones = rich_multiomics.get_node_pheromone()
        node_relevances = rich_multiomics.get_node_relevance()
        edge_pheromones = rich_multiomics.get_edge_pheromone()

        if cfg.RESULT.SAVE_RICH_DATA:
            save_object = [node_pheromones, edge_pheromones]
            save_data_name = cfg.RESULT.SAVE_RICH_DATA_TMPL.format(dataset_name=cfg.DATASET.NAME, fold_idx=fold_idx + 1)
            with gzip.open(os.path.join(cfg.RESULT.SAVE_RICH_DATA_DIR, save_data_name), 'wb') as file:
                pickle.dump(save_object, file)

        for feat_size in cfg.ACO.FINAL_FEAT_SIZES:
            print(f"  ==> Fold {fold_idx + 1} - feature size {feat_size}...")
            rich_multiomics_deepcopy = copy.deepcopy(rich_multiomics)
            start_time = time.time()
            final_feat_subset = select_top_feats(node_pheromones, node_relevances, feat_size, multiomics.num_omics,
                                                 cfg.ACO.SELECTION_RATE)
            end_time = time.time()
            running_time += end_time - start_time

            rich_multiomics_deepcopy.reduce_dimensionality(final_feat_subset)
            final_train_data = rich_multiomics_deepcopy.concatenate_data(is_train=True)
            final_test_data = rich_multiomics_deepcopy.concatenate_data(is_train=False)

            models = [
                RandomForestClassifier(),
                XGBClassifier(),
                KNeighborsClassifier(),
                MLPClassifier(max_iter=500),
                RidgeClassifier()
            ]

            for model in models:
                model_result = evaluate_model(
                    model,
                    train_data=final_train_data.values,
                    test_data=final_test_data.values,
                    train_label=rich_multiomics_deepcopy.get(0).train_label.values.ravel(),
                    test_label=rich_multiomics_deepcopy.get(0).test_label.values.ravel(),
                    num_classes=rich_multiomics_deepcopy.num_classes
                )

                for metric_name, metric_value in model_result.items():
                    cls_results[feat_size][model.__class__.__name__][metric_name].append(metric_value)
                    if cfg.RESULT.SAVE_RESULT:
                        result = [feat_size, model.__class__.__name__, metric_name, fold_idx + 1, metric_value]
                        save_output(output_file, result)

            time_results[feat_size].append(running_time)
            if cfg.RESULT.SAVE_RESULT:
                time_result = [feat_size, fold_idx + 1, running_time]
                save_output(output_file_time, time_result)

    if cfg.RESULT.SAVE_RESULT:
        sort_file(output_file, sorted_output_file, by=cfg.RESULT.FILE_HEADER_CLF[0:4])
        sort_file(output_file_time, sorted_output_file_time, by=cfg.RESULT.FILE_HEADER_TIME[0:2])

    print(f"\n==> Showing results...")
    for feat_size, models in cls_results.items():
        exe_time = round(np.mean(time_results[feat_size]) / min(cfg.DATASET.NUM_FOLDS, fold_max_workers))
        print(f"Feature size {feat_size} (execution time: {exe_time} seconds)")
        for model_name, metrics in models.items():
            print(f"    {model_name}:")
            for metric_name, values in metrics.items():
                average = np.mean(values)
                std = np.std(values)
                print(f"        - {metric_name}: {average:.3f}±{std:.3f}")


if __name__ == '__main__':
    main()
