import argparse
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, confusion_matrix, auc

import mkmmd_raytune
import networks
import utils


def get_best_threshold(scores, labels, metric):
    thresholds = np.arange(0, 1.01, 0.01)
    best_threshold = 0
    best_metric_score = 0
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        confusion_mat = confusion_matrix(labels, predictions)
        current_metric_score = utils.metric_from_confusion_matrix(confusion_mat, metric)
        if current_metric_score > best_metric_score:
            best_metric_score = current_metric_score
            best_threshold = threshold
    return best_threshold, best_metric_score


def get_cm_from_best_metric_threshold(val_scores, val_labels,
                                      test_scores, test_labels,
                                      metric):
    thr, _ = get_best_threshold(val_scores.detach().numpy().flatten(), val_labels, metric)
    return confusion_matrix(test_labels, [int(x > thr) for x in test_scores])


def save_results(split_type,
                 method_name,
                 dataset1,
                 dataset2=None,
                 accuracy=None, auc=None, f1=None, mcc=None):
    global results_df
    if split_type == 'loso':
        results_df = pd.concat((
            results_df,
            pd.DataFrame.from_dict({
                "Method": [method_name],
                "Dataset": [dataset1],
                "Accuracy": [accuracy],
                "AUC": [auc],
                "F1": [f1],
                "MCC": [mcc],
            })), ignore_index=True)
    elif split_type == 'toso':
        results_df = pd.concat((
            results_df,
            pd.DataFrame.from_dict({
                "Method": [method_name],
                "Training Dataset": [dataset1],
                "Testing Dataset": [dataset2],
                "Accuracy": [accuracy],
                "AUC": [auc],
                "F1": [f1],
                "MCC": [mcc],
            })), ignore_index=True)
    elif split_type == 'kfold':
        results_df = pd.concat((
            results_df,
            pd.DataFrame.from_dict({
                "Method": [method_name],
                "Dataset": [dataset1],
                "Accuracy": [accuracy],
                "AUC": [auc],
                "F1": [f1],
                "MCC": [mcc],
            })), ignore_index=True)


def get_metaDAN_results(disease, split_type, use_threshold=False):
    if split_type == 'kfold':
        pattern = rf"^{split_type}_{disease}_.+\.pt$"
        for filename in os.listdir('dan_results'):
            if re.match(pattern, filename):
                filepath = os.path.join('dan_results', filename)
                ray_res = torch.load(filepath, weights_only=False)
                if disease == 'crc':
                    data, meta = utils.load_CRC_data2(studies=[ray_res[0]['test_dataset']])
                elif disease == 'ibd':
                    data, meta = utils.load_IBD_data(studies=[ray_res[0]['test_dataset']])
                elif disease == 't2d':
                    data, meta = utils.load_T2D_data(studies=[ray_res[0]['test_dataset']])
                for i in range(len(ray_res)):
                    if use_threshold:
                        train_set, test_set, val_set = mkmmd_raytune.load_data_for_training(
                            data, meta, ray_res[i]['train_idx'], ray_res[i]['test_idx'], val_idx=ray_res[i]['val_idx'])
                        net = networks.DAN(
                            dim=ray_res[i]['best_state_dict']['hidden_layers.0.block.0.weight'].shape[-1],
                            dropout_rate=ray_res[i]['best_params']['dropout_rate'],
                            num_kernels=ray_res[i]['best_params']['num_kernels'],
                            out_dim=1,
                            num_hidden_layers=ray_res[i]['best_params']['num_hidden_layers'],
                            embed_size=ray_res[i]['best_params']['embed_size'],
                            num_mmd_layers=ray_res[i]['best_params']['num_mmd_layers'])

                        net.load_state_dict(ray_res[i]["best_state_dict"])
                        net.eval()
                        #  get scores and labels of validation and determine threshold
                        _, val_scores = net(torch.stack([val_set[j][0] for j in range(len(val_set))]))
                        val_scores = torch.sigmoid(val_scores)
                        val_labels = torch.stack([val_set[j][1] for j in range(len(val_set))]).numpy()
                        #  get scores and labels of testing data and use threshold to determine cm
                        _, test_scores = net(torch.stack([test_set[j][0] for j in range(len(test_set))]))
                        test_scores = torch.sigmoid(test_scores)
                        test_labels = torch.stack([test_set[j][1] for j in range(len(test_set))]).numpy()
                        save_results(
                            split_type,
                            'MetaDAN',
                            ray_res[i]['test_dataset'],
                            None,
                            utils.metric_from_confusion_matrix(
                                get_cm_from_best_metric_threshold(
                                    val_scores, val_labels, test_scores, test_labels, 'acc'),
                                metric='acc'),
                            ray_res[i]['test_auc'],
                            utils.metric_from_confusion_matrix(
                                get_cm_from_best_metric_threshold(
                                    val_scores, val_labels, test_scores, test_labels, 'f1'),
                                metric='f1'),
                            utils.metric_from_confusion_matrix(
                                get_cm_from_best_metric_threshold(
                                    val_scores, val_labels, test_scores, test_labels, 'mcc'),
                                metric='mcc')
                        )
                    else:
                        cm = ray_res[i]['test_confusion_matrix']
                        save_results(
                            split_type,
                            'MetaDAN',
                            ray_res[i]['test_dataset'],
                            None,
                            utils.metric_from_confusion_matrix(cm, metric='acc'),
                            ray_res[i]['test_auc'],
                            utils.metric_from_confusion_matrix(cm, metric='f1'),
                            utils.metric_from_confusion_matrix(cm, metric='mcc')
                        )

    elif split_type == 'loso':
        pattern = rf"^{split_type}_{disease}_.+\.pt$"
        if use_threshold:
            if disease == 'crc':
                data, meta = utils.load_CRC_data2()
            elif disease == 'ibd':
                data, meta = utils.load_IBD_data()
            elif disease == 't2d':
                data, meta = utils.load_T2D_data()

        for filename in os.listdir('dan_results'):
            if re.match(pattern, filename):
                filepath = os.path.join('dan_results', filename)
                ray_res = torch.load(filepath, weights_only=False)
                for i in range(len(ray_res)):
                    if use_threshold:
                        train_set, test_set, val_set = mkmmd_raytune.load_data_for_training(
                            data, meta, ray_res[i]['train_idx'], ray_res[i]['test_idx'], val_idx=ray_res[i]['val_idx'])
                        net = networks.DAN(
                            dim=ray_res[i]['best_state_dict']['hidden_layers.0.block.0.weight'].shape[-1],
                            dropout_rate=ray_res[i]['best_params']['dropout_rate'],
                            num_kernels=ray_res[i]['best_params']['num_kernels'],
                            out_dim=1,
                            num_hidden_layers=ray_res[i]['best_params']['num_hidden_layers'],
                            embed_size=ray_res[i]['best_params']['embed_size'],
                            num_mmd_layers=ray_res[i]['best_params']['num_mmd_layers'])

                        net.load_state_dict(ray_res[i]["best_state_dict"])
                        net.eval()
                        #  get scores and labels of validation and determine threshold
                        _, val_scores = net(torch.stack([val_set[j][0] for j in range(len(val_set))]))
                        val_scores = torch.sigmoid(val_scores)
                        val_labels = torch.stack([val_set[j][1] for j in range(len(val_set))]).numpy()
                        #  get scores and labels of testing data and use threshold to determine cm
                        test_embeds, test_scores = net(torch.stack([test_set[j][0] for j in range(len(test_set))]))
                        test_scores = torch.sigmoid(test_scores)
                        test_labels = torch.stack([test_set[j][1] for j in range(len(test_set))]).numpy()
                        save_results(
                            split_type,
                            'MetaDAN',
                            ray_res[i]['test_dataset'],
                            None,
                            utils.metric_from_confusion_matrix(
                                get_cm_from_best_metric_threshold(
                                    val_scores, val_labels, test_scores, test_labels, 'acc'),
                                metric='acc'),
                            ray_res[i]['test_auc'],
                            utils.metric_from_confusion_matrix(
                                get_cm_from_best_metric_threshold(
                                    val_scores, val_labels, test_scores, test_labels, 'f1'),
                                metric='f1'),
                            utils.metric_from_confusion_matrix(
                                get_cm_from_best_metric_threshold(
                                    val_scores, val_labels, test_scores, test_labels, 'mcc'),
                                metric='mcc')
                        )

                    else:
                        cm = ray_res[i]['test_confusion_matrix']
                        save_results(
                            split_type,
                            'MetaDAN',
                            ray_res[i]['test_dataset'],
                            None,
                            utils.metric_from_confusion_matrix(cm, metric='acc'),
                            ray_res[i]['test_auc'],
                            utils.metric_from_confusion_matrix(cm, metric='f1'),
                            utils.metric_from_confusion_matrix(cm, metric='mcc')
                        )
                    ...
    elif split_type == 'toso':
        pattern = rf"^{split_type}_{disease}_.+_train_.+_test\.pt$"
        for filename in os.listdir('dan_results'):
            if re.match(pattern, filename):
                filepath = os.path.join('dan_results', filename)
                ray_res = torch.load(filepath, weights_only=False)
                if use_threshold:
                    if disease == 'crc':
                        data, meta = utils.load_CRC_data2(studies=[ray_res[0]['test_dataset'],
                                                                   ray_res[0]['train_dataset']])
                    elif disease == 'ibd':
                        data, meta = utils.load_IBD_data(studies=[ray_res[0]['test_dataset'],
                                                                  ray_res[0]['train_dataset']])
                    elif disease == 't2d':
                        data, meta = utils.load_T2D_data(studies=[ray_res[0]['test_dataset'],
                                                                  ray_res[0]['train_dataset']])
                for i in range(len(ray_res)):
                    if use_threshold:
                        train_set, test_set, val_set = mkmmd_raytune.load_data_for_training(
                            data, meta, ray_res[i]['train_idx'], ray_res[i]['test_idx'], val_idx=ray_res[i]['val_idx'])
                        net = networks.DAN(
                            dim=ray_res[i]['best_state_dict']['hidden_layers.0.block.0.weight'].shape[-1],
                            dropout_rate=ray_res[i]['best_params']['dropout_rate'],
                            num_kernels=ray_res[i]['best_params']['num_kernels'],
                            out_dim=1,
                            num_hidden_layers=ray_res[i]['best_params']['num_hidden_layers'],
                            embed_size=ray_res[i]['best_params']['embed_size'],
                            num_mmd_layers=ray_res[i]['best_params']['num_mmd_layers'])

                        net.load_state_dict(ray_res[i]["best_state_dict"])
                        net.eval()
                        #  get scores and labels of validation and determine threshold
                        _, val_scores = net(torch.stack([train_set[j][0] for j in range(len(train_set))]))
                        val_scores = torch.sigmoid(val_scores)
                        val_labels = torch.stack([train_set[j][1] for j in range(len(train_set))]).numpy()
                        # _, val_scores = net(torch.stack([val_set[j][0] for j in range(len(val_set))]))
                        # val_scores = torch.sigmoid(val_scores)
                        # val_labels = torch.stack([val_set[j][1] for j in range(len(val_set))]).numpy()
                        #  get scores and labels of testing data and use threshold to determine cm
                        _, test_scores = net(torch.stack([test_set[j][0] for j in range(len(test_set))]))
                        test_scores = torch.sigmoid(test_scores)
                        test_labels = torch.stack([test_set[j][1] for j in range(len(test_set))]).numpy()
                        save_results(
                            split_type,
                            'MetaDAN',
                            ray_res[i]['train_dataset'],
                            ray_res[i]['test_dataset'],
                            utils.metric_from_confusion_matrix(
                                get_cm_from_best_metric_threshold(
                                    val_scores, val_labels, test_scores, test_labels, 'acc'),
                                metric='acc'),
                            ray_res[i]['test_auc'],
                            utils.metric_from_confusion_matrix(
                                get_cm_from_best_metric_threshold(
                                    val_scores, val_labels, test_scores, test_labels, 'f1'),
                                metric='f1'),
                            utils.metric_from_confusion_matrix(
                                get_cm_from_best_metric_threshold(
                                    val_scores, val_labels, test_scores, test_labels, 'mcc'),
                                metric='mcc')
                        )
                    else:
                        cm = ray_res[i]['test_confusion_matrix']
                        save_results(
                            split_type,
                            'MetaDAN',
                            ray_res[i]['train_dataset'],
                            ray_res[i]['test_dataset'],
                            utils.metric_from_confusion_matrix(cm, metric='acc'),
                            ray_res[i]['test_auc'],
                            utils.metric_from_confusion_matrix(cm, metric='f1'),
                            utils.metric_from_confusion_matrix(cm, metric='mcc')
                        )


def get_SIAMCAT_results(disease, split_type):
    if split_type == 'kfold':
        pattern = rf"^{split_type}_{disease}_.+\.csv$"
        for filename in os.listdir('SIAMCAT_results'):
            if re.match(pattern, filename):
                filepath = os.path.join('SIAMCAT_results', filename)
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if line.startswith("run/fold"):
                            ...
                        elif line.startswith("true labels"):
                            true_labels = [int(int(x) > 0) for x in line.split()[2:]]
                        elif line.startswith("estimated labels"):
                            estimated_labels = [int(x) for x in line.split()[2:]]
                        elif line.startswith("estimated probabilities"):
                            estimated_probs = [float(x) for x in line.split()[2:]]
                        elif line.startswith("sample index"):
                            # sample_ids.extend([x for x in line.split()[2:]])
                            cm = confusion_matrix(true_labels, estimated_labels)
                            if len(np.unique(true_labels)) > 1:
                                save_results(
                                    split_type,
                                    'SIAMCAT',
                                    filename.split('_', 2)[-1].rsplit('.', 1)[0],
                                    None,
                                    utils.metric_from_confusion_matrix(cm, metric='acc'),
                                    roc_auc_score(true_labels, estimated_probs),
                                    utils.metric_from_confusion_matrix(cm, metric='f1'),
                                    utils.metric_from_confusion_matrix(cm, metric='mcc')
                                )
                            else:
                                save_results(
                                    split_type,
                                    'SIAMCAT',
                                    filename.split('_', 2)[-1].rsplit('.', 1)[0],
                                    None,
                                    utils.metric_from_confusion_matrix(cm, metric='acc'),
                                    None,
                                    None,
                                    None
                                )

    elif split_type == 'loso':
        pattern = rf"^{split_type}_{disease}_.+\.csv$"
        for filename in os.listdir('SIAMCAT_results'):
            if re.match(pattern, filename):
                filepath = os.path.join('SIAMCAT_results', filename)
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if line.startswith("run/fold"):
                            ...
                        elif line.startswith("true labels"):
                            true_labels = [int(int(x) > 0) for x in line.split()[2:]]
                        elif line.startswith("estimated labels"):
                            estimated_labels = [int(x) for x in line.split()[2:]]
                        elif line.startswith("estimated probabilities"):
                            estimated_probs = [float(x) for x in line.split()[2:]]
                        elif line.startswith("sample index"):
                            # sample_ids.extend([x for x in line.split()[2:]])
                            cm = confusion_matrix(true_labels, estimated_labels)
                            if len(np.unique(true_labels)) > 1:
                                save_results(
                                    split_type,
                                    'SIAMCAT',
                                    filename.split('_', 2)[-1].rsplit('.', 1)[0],
                                    None,
                                    utils.metric_from_confusion_matrix(cm, metric='acc'),
                                    roc_auc_score(true_labels, estimated_probs),
                                    utils.metric_from_confusion_matrix(cm, metric='f1'),
                                    utils.metric_from_confusion_matrix(cm, metric='mcc')
                                )
                            else:
                                save_results(
                                    split_type,
                                    'SIAMCAT',
                                    filename.split('_', 2)[-1].rsplit('.', 1)[0],
                                    None,
                                    utils.metric_from_confusion_matrix(cm, metric='acc'),
                                    None,
                                    None,
                                    None
                                )

    elif split_type == 'toso':
        pattern = rf"^{split_type}_{disease}_.+_train_.+_test\.csv$"
        for filename in os.listdir('SIAMCAT_results'):
            if re.match(pattern, filename):
                filepath = os.path.join('SIAMCAT_results', filename)
                start_1 = filename.find('_', filename.find('_') + 1) + 1  # After the second underscore
                end_1 = filename.find('_train')  # Before "_train"
                start_2 = filename.find('_', end_1 + 1) + 1  # After "_train"
                end_2 = filename.find('_test')  # Before "_test"
                train_study = filename[start_1:end_1]
                test_study = filename[start_2:end_2]
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if line.startswith("run/fold"):
                            ...
                        elif line.startswith("true labels"):
                            true_labels = [int(int(x) > 0) for x in line.split()[2:]]
                        elif line.startswith("estimated labels"):
                            estimated_labels = [int(x) for x in line.split()[2:]]
                        elif line.startswith("estimated probabilities"):
                            estimated_probs = [float(x) for x in line.split()[2:]]
                        elif line.startswith("sample index"):
                            # sample_ids.extend([x for x in line.split()[2:]])
                            cm = confusion_matrix(true_labels, estimated_labels)
                            if len(np.unique(true_labels)) > 1:
                                save_results(
                                    split_type,
                                    'SIAMCAT',
                                    train_study,
                                    test_study,
                                    utils.metric_from_confusion_matrix(cm, metric='acc'),
                                    roc_auc_score(true_labels, estimated_probs),
                                    utils.metric_from_confusion_matrix(cm, metric='f1'),
                                    utils.metric_from_confusion_matrix(cm, metric='mcc')
                                )
                            else:
                                save_results(
                                    split_type,
                                    'SIAMCAT',
                                    train_study,
                                    test_study,
                                    utils.metric_from_confusion_matrix(cm, metric='acc'),
                                    None,
                                    None,
                                    None
                                )


def get_metAML_results(disease, split_type):
    if split_type == 'kfold':
        pattern = rf"^{split_type}_{disease}_.+_estimations\.txt$"
        for filename in os.listdir('MetAML_results'):
            if re.match(pattern, filename):
                filepath = os.path.join('MetAML_results', filename)
                start = filename.find('_', filename.find('_') + 1) + 1
                end = filename.rfind('_estimations')
                study_name = filename[start:end]

                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if line.startswith("run/fold"):
                            ...
                        elif line.startswith("true labels"):
                            true_labels = [int(int(x) > 0) for x in line.split()[2:]]
                        elif line.startswith("estimated labels"):
                            estimated_labels = [int(x) for x in line.split()[2:]]
                        elif line.startswith("estimated probabilities"):
                            estimated_probs = [float(x) for x in line.split()[2:]]
                        elif line.startswith("sample index"):
                            # sample_ids.extend([x for x in line.split()[2:]])
                            cm = confusion_matrix(true_labels, estimated_labels)
                            if len(np.unique(true_labels)) > 1:
                                save_results(
                                    split_type,
                                    'MetAML',
                                    study_name,
                                    None,
                                    utils.metric_from_confusion_matrix(cm, metric='acc'),
                                    roc_auc_score(true_labels, estimated_probs),
                                    utils.metric_from_confusion_matrix(cm, metric='f1'),
                                    utils.metric_from_confusion_matrix(cm, metric='mcc')
                                )
                            else:
                                save_results(
                                    split_type,
                                    'MetAML',
                                    study_name,
                                    None,
                                    utils.metric_from_confusion_matrix(cm, metric='acc'),
                                    None,
                                    None,
                                    None
                                )
                            ...

    elif split_type == 'loso':
        pattern = rf"^{split_type}_{disease}_.+_estimations\.txt$"
        for filename in os.listdir('MetAML_results'):
            if re.match(pattern, filename):
                filepath = os.path.join('MetAML_results', filename)
                start = filename.find('_', filename.find('_') + 1) + 1
                end = filename.rfind('_estimations')
                study_name = filename[start:end]

                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if line.startswith("run/fold"):
                            ...
                        elif line.startswith("true labels"):
                            true_labels = [int(int(x) > 0) for x in line.split()[2:]]
                        elif line.startswith("estimated labels"):
                            estimated_labels = [int(x) for x in line.split()[2:]]
                        elif line.startswith("estimated probabilities"):
                            estimated_probs = [float(x) for x in line.split()[2:]]
                        elif line.startswith("sample index"):
                            # sample_ids.extend([x for x in line.split()[2:]])
                            cm = confusion_matrix(true_labels, estimated_labels)
                            if len(np.unique(true_labels)) > 1:
                                save_results(
                                    split_type,
                                    'MetAML',
                                    study_name,
                                    None,
                                    utils.metric_from_confusion_matrix(cm, metric='acc'),
                                    roc_auc_score(true_labels, estimated_probs),
                                    utils.metric_from_confusion_matrix(cm, metric='f1'),
                                    utils.metric_from_confusion_matrix(cm, metric='mcc')
                                )
                            else:
                                save_results(
                                    split_type,
                                    'MetAML',
                                    study_name,
                                    None,
                                    utils.metric_from_confusion_matrix(cm, metric='acc'),
                                    None,
                                    None,
                                    None
                                )
    elif split_type == 'toso':
        pattern = rf"^{split_type}_{disease}_.+_train_.+_test_estimations\.txt$"
        for filename in os.listdir('MetAML_results'):
            if re.match(pattern, filename):
                filepath = os.path.join('MetAML_results', filename)
                start_1 = filename.find('_', filename.find('_') + 1) + 1  # After the second underscore
                end_1 = filename.find('_train')  # Before "_train"
                start_2 = filename.find('_', end_1 + 1) + 1  # After "_train"
                end_2 = filename.find('_test')  # Before "_test"
                train_study = filename[start_1:end_1]
                test_study = filename[start_2:end_2]
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if line.startswith("run/fold"):
                            ...
                        elif line.startswith("true labels"):
                            true_labels = [int(int(x) > 0) for x in line.split()[2:]]
                        elif line.startswith("estimated labels"):
                            estimated_labels = [int(x) for x in line.split()[2:]]
                        elif line.startswith("estimated probabilities"):
                            estimated_probs = [float(x) for x in line.split()[2:]]
                        elif line.startswith("sample index"):
                            # sample_ids.extend([x for x in line.split()[2:]])
                            cm = confusion_matrix(true_labels, estimated_labels)
                            if len(np.unique(true_labels)) > 1:
                                save_results(
                                    split_type,
                                    'MetAML',
                                    train_study,
                                    test_study,
                                    utils.metric_from_confusion_matrix(cm, metric='acc'),
                                    roc_auc_score(true_labels, estimated_probs),
                                    utils.metric_from_confusion_matrix(cm, metric='f1'),
                                    utils.metric_from_confusion_matrix(cm, metric='mcc')
                                )
                            else:
                                save_results(
                                    split_type,
                                    'MetAML',
                                    train_study,
                                    test_study,
                                    utils.metric_from_confusion_matrix(cm, metric='acc'),
                                    None,
                                    None,
                                    None
                                )
    ...


def compute_mean_and_me(df,
                        split_type,
                        confidence_level=0.95,
                        mean_only=False):
    """
    Computes the mean and margin of error (ME) for specified metric columns, grouped by the specified columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        confidence_level (float): Confidence level for the t-distribution (default is 0.95).

    Returns:
        pd.DataFrame: A DataFrame containing the mean(ME) for each group and metric.
    """
    if split_type == 'toso':
        group_cols = ['Method', 'Training Dataset', 'Testing Dataset']
        metric_cols = ['Accuracy', 'AUC', 'F1', 'MCC']
    else:
        group_cols = ['Method', 'Dataset']
        metric_cols = ['Accuracy', 'AUC', 'F1', 'MCC']

    def margin_of_error(data, confidence_level):
        n = len(data)
        if n <= 1:  # Avoid division by zero or invalid t-critical computation
            return 0
        t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, df=n - 1)
        return t_critical * (np.std(data, ddof=1) / np.sqrt(n))

    if mean_only:
        result = (
            df.groupby(group_cols)
            .apply(lambda group: pd.Series(
                {col: np.mean(group[col]) for col in
                 metric_cols}
            ))
            .reset_index()
        )
    else:
        result = (
            df.groupby(group_cols)
            .apply(lambda group: pd.Series(
                {col: f"{np.mean(group[col]):.4f}({margin_of_error(group[col], confidence_level):.4f})" for col in
                 metric_cols}
            ))
            .reset_index()
        )

    return result


def generate_heatmaps(df, split_type, output_folder):
    metrics = ['AUC', 'F1', 'MCC']

    for metric in metrics:

        for method in df['Method'].unique():
            temp_df = df[df['Method'] == method]
            # Pivot the dataframe to create a matrix format for heatmap
            heatmap_data = temp_df.pivot_table(
                index='Training Dataset',
                columns='Testing Dataset',
                values=metric
            )

            # Plot heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(heatmap_data, annot=True, cmap='rocket', vmin=0, vmax=1,
                        fmt='.3f', cbar_kws={'label': metric})
            plt.title(f'{method} ({metric})')
            plt.xlabel('Testing Dataset')
            plt.ylabel('Training Dataset')
            # Save the heatmap as a PNG file
            plt.savefig(f'{output_folder}/{split_type}_heatmap_{metric}_{method}.png', dpi=300)
            plt.show()
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model comparisons')
    parser.add_argument('--split_type', type=str, default='kfold', help='kfold/loso/toso')
    parser.add_argument('--disease', type=str, default='crc', help='crc/ibd/t2d')

    args = parser.parse_args()
    split_type = args.split_type
    disease = args.disease

    output_folder = "./comparison_figures"
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    if split_type == 'toso':
        results_df = pd.DataFrame(columns=["Method",
                                           "Training Dataset", "Testing Dataset",
                                           "Accuracy", "AUC", "F1", "MCC"])
    else:
        results_df = pd.DataFrame(columns=["Method",
                                           "Dataset",
                                           "Accuracy", "AUC", "F1", "MCC"])
    # get_metaDAN_results(disease, split_type)
    get_metaDAN_results(disease, split_type, use_threshold=True)
    get_SIAMCAT_results(disease, split_type)
    get_metAML_results(disease, split_type)
    if split_type == 'toso':
        for d in results_df['Training Dataset'].unique():
            results_df.loc[results_df['Training Dataset'] == d, 'Training Dataset'] = d.capitalize()
            results_df.loc[results_df['Testing Dataset'] == d, 'Testing Dataset'] = d.capitalize()
    else:
        for d in results_df['Dataset'].unique():
            results_df.loc[results_df['Dataset'] == d, 'Dataset'] = d.capitalize()
    t = compute_mean_and_me(results_df, split_type, mean_only=False)
    ...
    generate_heatmaps(results_df, split_type, output_folder)

    ...
