import os
import csv
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightning.pytorch as pl
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix


def seed_everything(seed=None, workers=False):
    # Pytorch lightning
    pl.seed_everything(seed=seed, workers=workers)

    # Pytorch
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset_indices(split_folder):
    train_file = os.path.join(split_folder, "train_index.csv")
    test_file = os.path.join(split_folder, "test_index.csv")

    train_index = pd.read_csv(train_file)["Index"].tolist()
    test_index = pd.read_csv(test_file)["Index"].tolist()

    return train_index, test_index


def create_file(file_dir, header):
    with open(file_dir, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)


def save_output(file_dir, output, multi_rows=False):
    with open(file_dir, mode='a', newline='') as file:
        writer = csv.writer(file)
        if multi_rows:
            writer.writerows(output)
        else:
            writer.writerow(output)


def sort_file(origin_dir, sorted_dir, by):
    df = pd.read_csv(origin_dir)
    df = df.sort_values(by=by)
    df.to_csv(sorted_dir, index=False)


def is_directory_empty(directory):
    if os.path.exists(directory):
        return not os.listdir(directory)
    return True


def normalize(df, new_min=0.0, new_max=1.0):
    if isinstance(df, list):
        df_array = np.array(df)
        return ((df_array - df_array.min()) * (new_max - new_min) / (df_array.max() - df_array.min())) + new_min

    # apply the min-max scaling for each feature separately
    return ((df - df.min()) * (new_max - new_min) / (df.max() - df.min())) + new_min


class RelevanceMetric(Enum):
    ANOVA = "ANOVA"


class CorrelationMetric(Enum):
    PEARSON_CORRELATION = "PEARSON CORRELATION"


class ClassifierName(Enum):
    RANDOM_FOREST = "Random Forest"
    SVM = "Support Vector Machine"


def compute_relevance(dataframe, class_label=None, rel_metric: RelevanceMetric = RelevanceMetric.ANOVA):
    match rel_metric:
        case RelevanceMetric.ANOVA:
            f_values, _ = f_classif(dataframe, class_label.squeeze())
            return pd.Series(f_values, index=dataframe.columns)
        case _:
            raise Exception("This metric is not still implemented")


def compute_correlation(dataframe, cor_metric: CorrelationMetric = CorrelationMetric.PEARSON_CORRELATION):
    match cor_metric:
        case CorrelationMetric.PEARSON_CORRELATION:
            correlation = dataframe.corr(method='pearson')
            correlation = correlation.map(abs)
            return correlation
        case _:
            raise Exception("This metric is not still implemented")


def evaluate_feat_subset(training_data, training_label, num_classes, classifier=ClassifierName.SVM):
    match classifier:
        case ClassifierName.RANDOM_FOREST:
            algorithm = RandomForestClassifier()
        case ClassifierName.SVM:
            algorithm = SVC(kernel='linear', probability=True)
        case _:
            raise ValueError("This classifier is not still implemented")

    scoring_method = 'f1' if num_classes == 2 else 'f1_macro'
    metric_val = cross_val_score(algorithm, training_data, training_label, cv=5, scoring=scoring_method)

    return metric_val.mean()


def evaluate_model(model, train_data, test_data, train_label, test_label, num_classes=2):
    model.fit(train_data, train_label)
    predicted_results = model.predict(test_data)

    metrics_dict = {}  # Dictionary to store the metrics

    if num_classes == 2:
        acc = accuracy_score(test_label, predicted_results)
        auc = roc_auc_score(test_label, predicted_results)
        sensitivity, specificity, ppv, npv = calculate_performance_metrics(test_label, predicted_results)

        metrics_dict['AUROC'] = auc
        metrics_dict['Accuracy'] = acc
        metrics_dict['NPV'] = npv
        metrics_dict['PPV'] = ppv
        metrics_dict['Sensitivity'] = sensitivity
        metrics_dict['Specificity'] = specificity
    else:
        acc = accuracy_score(test_label, predicted_results)
        f1_macro = f1_score(test_label, predicted_results, average='macro')
        f1_micro = f1_score(test_label, predicted_results, average='micro')
        f1_weighted = f1_score(test_label, predicted_results, average='weighted')
        precision = precision_score(test_label, predicted_results, average="weighted")
        recall = recall_score(test_label, predicted_results, average="weighted")

        metrics_dict['Accuracy'] = acc
        metrics_dict['F1_macro'] = f1_macro
        metrics_dict['F1_micro'] = f1_micro
        metrics_dict['F1_weighted'] = f1_weighted
        metrics_dict['Precision'] = precision
        metrics_dict['Recall'] = recall

    return metrics_dict


def select_top_feats(pheromone, relevance, num_top_feats, num_modalities=3, selection_rate=0.5):
    quotas = [round(num_top_feats / num_modalities) for _ in range(len(pheromone))]
    diff = num_top_feats - sum(quotas)
    quotas[-1] += diff

    selected_indices = {}

    for idx, (pheromone_sublist, relevance_sublist, quota) in enumerate(zip(pheromone, relevance, quotas)):
        pheromone_sublist = normalize(pheromone_sublist, 0.1, 1.0)
        average_values = []
        for p, r in zip(pheromone_sublist, relevance_sublist):
            average_value = ((selection_rate * p) + ((1.0 - selection_rate) * r))
            average_values.append(average_value)

        top_indices = sorted(range(len(average_values)), key=lambda i: average_values[i], reverse=True)[:quota]
        selected_indices[idx] = top_indices

    return selected_indices


def calculate_performance_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return sensitivity, specificity, ppv, npv


def xavier_init(module) -> None:
    if type(module) == nn.Linear:
        nn.init.xavier_normal_(module.weight)


def bias_init(module) -> None:
    if type(module) == nn.Linear and module.bias is not None:
        module.bias.data.fill_(0.0)
