import argparse
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib as mpl
from matplotlib.patches import Ellipse
from skbio.stats.ordination import pcoa
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.spatial.distance import squareform, pdist
from scipy.stats import sem
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, confusion_matrix, auc, roc_curve

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


def get_metaDAN_results(disease, split_type, transform, use_threshold=True):
    global auroc_dict
    global embedding_dict
    if split_type == 'kfold':
        pattern = rf"^{split_type}_{disease}_.+{transform}\.pt$"
        for filename in os.listdir('dan_results'):
            if re.match(pattern, filename):
                filepath = os.path.join('dan_results', filename)
                ray_res = torch.load(filepath, weights_only=False)
                if disease == 'crc':
                    data, meta = utils.load_CRC_data(studies_to_include=[ray_res[0]['test_dataset']],
                                                     transform=transform,
                                                     num_feat=150)
                elif disease == 'ibd':
                    data, meta = utils.load_IBD_data(studies_to_include=[ray_res[0]['test_dataset']],
                                                     transform=transform,
                                                     num_feat=150)
                elif disease == 't2d':
                    data, meta = utils.load_T2D_data(studies_to_include=[ray_res[0]['test_dataset']],
                                                     transform=transform,
                                                     num_feat=150)
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
                        if ray_res[i]['test_dataset'] in auroc_dict:
                            if 'MetaDAN' in auroc_dict[ray_res[i]['test_dataset']]:
                                auroc_dict[ray_res[i]['test_dataset']]['MetaDAN']['y_true'].append(
                                    test_labels.flatten())
                                auroc_dict[ray_res[i]['test_dataset']]['MetaDAN']['y_score'].append(
                                    test_scores.detach().numpy().flatten())
                            else:
                                auroc_dict[ray_res[i]['test_dataset']]['MetaDAN'] = {
                                    'y_true': [test_labels.flatten()],
                                    'y_score': [test_scores.detach().numpy().flatten()]
                                }
                        else:
                            auroc_dict[ray_res[i]['test_dataset']] = {}
                            auroc_dict[ray_res[i]['test_dataset']]['MetaDAN'] = {
                                'y_true': [test_labels.flatten()],
                                'y_score': [test_scores.detach().numpy().flatten()]
                            }

                        save_results(
                            split_type,
                            f'MetaDAN_{transform}',
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
                            f'MetaDAN_{transform}',
                            ray_res[i]['test_dataset'],
                            None,
                            utils.metric_from_confusion_matrix(cm, metric='acc'),
                            ray_res[i]['test_auc'],
                            utils.metric_from_confusion_matrix(cm, metric='f1'),
                            utils.metric_from_confusion_matrix(cm, metric='mcc')
                        )

    elif split_type == 'loso':
        pattern = rf"^{split_type}_{disease}_.+{transform}\.pt$"
        if use_threshold:
            if disease == 'crc':
                data, meta1 = utils.load_CRC_data()
            elif disease == 'ibd':
                data, meta1 = utils.load_IBD_data()
            elif disease == 't2d':
                data, meta1 = utils.load_T2D_data()

        for filename in os.listdir('dan_results'):
            if re.match(pattern, filename):
                filepath = os.path.join('dan_results', filename)
                ray_res = torch.load(filepath, weights_only=False)
                for i in range(len(ray_res)):
                    if use_threshold:
                        if disease == 'crc':
                            data, meta = utils.load_CRC_data(
                                transform=transform, num_feat=150,
                                train_studies=list(meta1.loc[meta1['Dataset'] != ray_res[i]['test_dataset']]['Dataset'].unique()))
                        elif disease == 'ibd':
                            data, meta = utils.load_IBD_data(
                                transform=transform, num_feat=150,
                                train_studies=list(meta1.loc[meta1['Dataset'] != ray_res[i]['test_dataset']]['Dataset'].unique()))
                        elif disease == 't2d':
                            data, meta = utils.load_T2D_data(
                                transform=transform, num_feat=150,
                                train_studies=list(meta1.loc[meta1['Dataset'] != ray_res[i]['test_dataset']]['Dataset'].unique()))

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
                        # get all embeddings for later use
                        X_train = torch.tensor(data.loc[ray_res[i]['train_idx']].values.astype(float),
                                               dtype=torch.float32)
                        s, mu = torch.std_mean(X_train, dim=0)
                        X = torch.tensor(data.values.astype(float), dtype=torch.float32)
                        X = (X - mu) / s
                        X = torch.nan_to_num(X, nan=0.0)
                        embeddings, _ = net(X)
                        embeddings = [embeddings[j].detach().numpy() for j in range(len(embeddings))]
                        if ray_res[i]['test_dataset'] in embedding_dict:
                            if ray_res[i]['test_auc'] > embedding_dict[ray_res[i]['test_dataset']]['test_auc']:
                                embedding_dict[ray_res[i]['test_dataset']]['test_auc'] = ray_res[i]['test_auc']
                                embedding_dict[ray_res[i]['test_dataset']]['embeddings'] = embeddings

                        else:
                            embedding_dict[ray_res[i]['test_dataset']] = {}
                            embedding_dict[ray_res[i]['test_dataset']]['test_auc'] = ray_res[i]['test_auc']
                            embedding_dict[ray_res[i]['test_dataset']]['embeddings'] = embeddings

                        #  get scores and labels of validation and determine threshold
                        _, val_scores = net(torch.stack([val_set[j][0] for j in range(len(val_set))]))
                        val_scores = torch.sigmoid(val_scores)
                        val_labels = torch.stack([val_set[j][1] for j in range(len(val_set))]).numpy()
                        #  get scores and labels of testing data and use threshold to determine cm
                        test_embeds, test_scores = net(torch.stack([test_set[j][0] for j in range(len(test_set))]))
                        test_scores = torch.sigmoid(test_scores)
                        test_labels = torch.stack([test_set[j][1] for j in range(len(test_set))]).numpy()
                        if ray_res[i]['test_dataset'] in auroc_dict:
                            if 'MetaDAN' in auroc_dict[ray_res[i]['test_dataset']]:
                                auroc_dict[ray_res[i]['test_dataset']]['MetaDAN']['y_true'].append(
                                    test_labels.flatten())
                                auroc_dict[ray_res[i]['test_dataset']]['MetaDAN']['y_score'].append(
                                    test_scores.detach().numpy().flatten())
                            else:
                                auroc_dict[ray_res[i]['test_dataset']]['MetaDAN'] = {
                                    'y_true': [test_labels.flatten()],
                                    'y_score': [test_scores.detach().numpy().flatten()]
                                }
                        else:
                            auroc_dict[ray_res[i]['test_dataset']] = {}
                            auroc_dict[ray_res[i]['test_dataset']]['MetaDAN'] = {
                                'y_true': [test_labels.flatten()],
                                'y_score': [test_scores.detach().numpy().flatten()]
                            }
                        save_results(
                            split_type,
                            f'MetaDAN_{transform}',
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
                            f'MetaDAN_{transform}',
                            ray_res[i]['test_dataset'],
                            None,
                            utils.metric_from_confusion_matrix(cm, metric='acc'),
                            ray_res[i]['test_auc'],
                            utils.metric_from_confusion_matrix(cm, metric='f1'),
                            utils.metric_from_confusion_matrix(cm, metric='mcc')
                        )
                    ...
    elif split_type == 'toso':
        pattern = rf"^{split_type}_{disease}_.+_train_.+_test_{transform}\.pt$"
        for filename in os.listdir('dan_results'):
            if re.match(pattern, filename):
                filepath = os.path.join('dan_results', filename)
                ray_res = torch.load(filepath, weights_only=False)
                if use_threshold:
                    if disease == 'crc':
                        data, meta = utils.load_CRC_data(studies_to_include=[ray_res[0]['test_dataset'],
                                                                             ray_res[0]['train_dataset']],
                                                         transform=transform,
                                                         num_feat=150,
                                                         train_studies=[ray_res[0]['train_dataset']])
                    elif disease == 'ibd':
                        data, meta = utils.load_IBD_data(studies_to_include=[ray_res[0]['test_dataset'],
                                                                             ray_res[0]['train_dataset']],
                                                         transform=transform,
                                                         num_feat=150,
                                                         train_studies=[ray_res[0]['train_dataset']])
                    elif disease == 't2d':
                        data, meta = utils.load_T2D_data(studies_to_include=[ray_res[0]['test_dataset'],
                                                                             ray_res[0]['train_dataset']],
                                                         transform=transform,
                                                         num_feat=150,
                                                         train_studies=[ray_res[0]['train_dataset']])
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
                        if ray_res[i]['train_dataset'] in auroc_dict:
                            if ray_res[i]['test_dataset'] in auroc_dict[ray_res[i]['train_dataset']]:
                                if 'MetaDAN' in auroc_dict[ray_res[i]['train_dataset']][ray_res[i]['test_dataset']]:
                                    auroc_dict[ray_res[i]['train_dataset']][ray_res[i]['test_dataset']]['MetaDAN'][
                                        'y_true'].append(test_labels.flatten())
                                    auroc_dict[ray_res[i]['train_dataset']][ray_res[i]['test_dataset']]['MetaDAN'][
                                        'y_score'].append(test_scores.detach().numpy().flatten())
                                else:
                                    auroc_dict[ray_res[i]['train_dataset']][ray_res[i]['test_dataset']]['MetaDAN'] = {
                                        'y_true': [test_labels.flatten()],
                                        'y_score': [test_scores.detach().numpy().flatten()]
                                    }
                            else:
                                auroc_dict[ray_res[i]['train_dataset']][ray_res[i]['test_dataset']] = {}
                                auroc_dict[ray_res[i]['train_dataset']][ray_res[i]['test_dataset']]['MetaDAN'] = {
                                    'y_true': [test_labels.flatten()],
                                    'y_score': [test_scores.detach().numpy().flatten()]
                                }
                        else:
                            auroc_dict[ray_res[i]['train_dataset']] = {}
                            auroc_dict[ray_res[i]['train_dataset']][ray_res[i]['test_dataset']] = {}
                            auroc_dict[ray_res[i]['train_dataset']][ray_res[i]['test_dataset']]['MetaDAN'] = {
                                'y_true': [test_labels.flatten()],
                                'y_score': [test_scores.detach().numpy().flatten()]
                            }
                        save_results(
                            split_type,
                            f'MetaDAN_{transform}',
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
                            f'MetaDAN_{transform}',
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
                            study_name = filename.split('_', 2)[-1].rsplit('.', 1)[0]
                            if study_name in auroc_dict:
                                if 'SIAMCAT' in auroc_dict[study_name]:
                                    auroc_dict[study_name]['SIAMCAT']['y_true'].append(
                                        np.array(true_labels))
                                    auroc_dict[study_name]['SIAMCAT']['y_score'].append(
                                        np.array(estimated_probs))
                                else:
                                    auroc_dict[study_name]['SIAMCAT'] = {
                                        'y_true': [np.array(true_labels)],
                                        'y_score': [np.array(estimated_probs)]
                                    }
                            else:
                                auroc_dict[study_name] = {}
                                auroc_dict[study_name]['SIAMCAT'] = {
                                    'y_true': [np.array(true_labels)],
                                    'y_score': [np.array(estimated_probs)]
                                }

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
                            study_name = filename.split('_', 2)[-1].rsplit('.', 1)[0]
                            if study_name in auroc_dict:
                                if 'SIAMCAT' in auroc_dict[study_name]:
                                    auroc_dict[study_name]['SIAMCAT']['y_true'].append(
                                        np.array(true_labels))
                                    auroc_dict[study_name]['SIAMCAT']['y_score'].append(
                                        np.array(estimated_probs))
                                else:
                                    auroc_dict[study_name]['SIAMCAT'] = {
                                        'y_true': [np.array(true_labels)],
                                        'y_score': [np.array(estimated_probs)]
                                    }
                            else:
                                auroc_dict[study_name] = {}
                                auroc_dict[study_name]['SIAMCAT'] = {
                                    'y_true': [np.array(true_labels)],
                                    'y_score': [np.array(estimated_probs)]
                                }
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
                            if train_study in auroc_dict:
                                if test_study in auroc_dict[train_study]:
                                    if 'SIAMCAT' in auroc_dict[train_study][test_study]:
                                        auroc_dict[train_study][test_study]['SIAMCAT'][
                                            'y_true'].append(np.array(true_labels))
                                        auroc_dict[train_study][test_study]['SIAMCAT'][
                                            'y_score'].append(np.array(estimated_probs))
                                    else:
                                        auroc_dict[train_study][test_study][
                                            'SIAMCAT'] = {
                                            'y_true': [np.array(true_labels)],
                                            'y_score': [np.array(estimated_probs)]
                                        }
                                else:
                                    auroc_dict[train_study][test_study] = {}
                                    auroc_dict[train_study][test_study]['SIAMCAT'] = {
                                        'y_true': [np.array(true_labels)],
                                        'y_score': [np.array(estimated_probs)]
                                    }
                            else:
                                auroc_dict[train_study] = {}
                                auroc_dict[train_study][test_study] = {}
                                auroc_dict[train_study][test_study]['SIAMCAT'] = {
                                    'y_true': [np.array(true_labels)],
                                    'y_score': [np.array(estimated_probs)]
                                }
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
                            if study_name in auroc_dict:
                                if 'MetAML' in auroc_dict[study_name]:
                                    auroc_dict[study_name]['MetAML']['y_true'].append(
                                        np.array(true_labels))
                                    auroc_dict[study_name]['MetAML']['y_score'].append(
                                        np.array(estimated_probs))
                                else:
                                    auroc_dict[study_name]['MetAML'] = {
                                        'y_true': [np.array(true_labels)],
                                        'y_score': [np.array(estimated_probs)]
                                    }
                            else:
                                auroc_dict[study_name] = {}
                                auroc_dict[study_name]['MetAML'] = {
                                    'y_true': [np.array(true_labels)],
                                    'y_score': [np.array(estimated_probs)]
                                }
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
                            if study_name in auroc_dict:
                                if 'MetAML' in auroc_dict[study_name]:
                                    auroc_dict[study_name]['MetAML']['y_true'].append(
                                        np.array(true_labels))
                                    auroc_dict[study_name]['MetAML']['y_score'].append(
                                        np.array(estimated_probs))
                                else:
                                    auroc_dict[study_name]['MetAML'] = {
                                        'y_true': [np.array(true_labels)],
                                        'y_score': [np.array(estimated_probs)]
                                    }
                            else:
                                auroc_dict[study_name] = {}
                                auroc_dict[study_name]['MetAML'] = {
                                    'y_true': [np.array(true_labels)],
                                    'y_score': [np.array(estimated_probs)]
                                }
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
                            if train_study in auroc_dict:
                                if test_study in auroc_dict[train_study]:
                                    if 'MetAML' in auroc_dict[train_study][test_study]:
                                        auroc_dict[train_study][test_study]['MetAML'][
                                            'y_true'].append(np.array(true_labels))
                                        auroc_dict[train_study][test_study]['MetAML'][
                                            'y_score'].append(np.array(estimated_probs))
                                    else:
                                        auroc_dict[train_study][test_study][
                                            'MetAML'] = {
                                            'y_true': [np.array(true_labels)],
                                            'y_score': [np.array(estimated_probs)]
                                        }
                                else:
                                    auroc_dict[train_study][test_study] = {}
                                    auroc_dict[train_study][test_study]['MetAML'] = {
                                        'y_true': [np.array(true_labels)],
                                        'y_score': [np.array(estimated_probs)]
                                    }
                            else:
                                auroc_dict[train_study] = {}
                                auroc_dict[train_study][test_study] = {}
                                auroc_dict[train_study][test_study]['MetAML'] = {
                                    'y_true': [np.array(true_labels)],
                                    'y_score': [np.array(estimated_probs)]
                                }
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
        :param split_type:
        :param confidence_level:
        :param mean_only:
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


def generate_heatmaps(df, split_type, disease, output_folder):
    metrics = ['AUC', 'F1', 'MCC']
    rename_dict = {
        'HanniganGD_2017': 'Hannigan',
        'FengQ_2015': 'Feng',
        'GuptaA_2019': 'Gupta',
        'YachidaS_2019': 'Yachida',
        'YuJ_2015': 'Yu',
        'ThomasAM_2019_c': 'Thomas',
        'ZellerG_2014': 'Zeller',
        'VogtmannE_2016': 'Vogtmann',
        'WirbelJ_2018': 'Wirbel',
        'NielsenHB_2014': 'Nielsen',
        'HMP_2019_ibdmdb': 'IBDMDB',
        'IjazUZ_2017': 'Ijaz',
        'KarlssonFH_2013': 'Karlsson',
        'QinJ_2012': 'Qin',
        'MetaDAN_clr': 'MetaDAN'

    }
    df = df.replace({"Training Dataset": rename_dict})
    df = df.replace({"Testing Dataset": rename_dict})
    df = df.replace({"Method": rename_dict})
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
            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap_data, annot=True, cmap='rocket', vmin=0, vmax=1,
                        fmt='.3f', cbar_kws={'label': metric}, annot_kws={'size': 13})
            plt.title(f'{method} ({metric})', fontsize=15)
            plt.xlabel('Testing Dataset', fontsize=15)
            plt.ylabel('Training Dataset', fontsize=15)
            plt.xticks(rotation=45, ha='center', fontsize=11)
            plt.yticks(rotation=0, ha='right', fontsize=11)
            plt.tight_layout()
            plt.savefig(f'{output_folder}/{split_type}_{disease}_heatmap_{metric}_{method}.png', dpi=300)
            plt.show()
            plt.close()


def plot_metrics_boxplots(df):
    metrics = ["Accuracy", "AUC", "F1", "MCC"]
    rename_dict = {
        'HanniganGD_2017': 'Hannigan (CRC)',
        'FengQ_2015': 'Feng (CRC)',
        'GuptaA_2019': 'Gupta (CRC)',
        'YachidaS_2019': 'Yachida (CRC)',
        'YuJ_2015': 'Yu (CRC)',
        'ThomasAM_2019_c': 'Thomas (CRC)',
        'ZellerG_2014': 'Zeller (CRC)',
        'VogtmannE_2016': 'Vogtmann (CRC)',
        'WirbelJ_2018': 'Wirbel (CRC)',
        'NielsenHB_2014': 'Nielsen (IBD)',
        'HMP_2019_ibdmdb': 'IBDMDB (IBD)',
        'IjazUZ_2017': 'Ijaz (IBD)',
        'KarlssonFH_2013': 'Karlsson (T2D)',
        'QinJ_2012': 'Qin (T2D)'
    }
    df = df.replace({"Dataset": rename_dict})
    dataset_order = [
        "Feng (CRC)",
        "Gupta (CRC)",
        "Hannigan (CRC)",
        "Thomas (CRC)",
        "Vogtmann (CRC)",
        "Wirbel (CRC)",
        "Yachida (CRC)",
        "Yu (CRC)",
        "Zeller (CRC)",
        "IBDMDB (IBD)",
        "Ijaz (IBD)",
        "Nielsen (IBD)",
        "Karlsson (T2D)",
        "Qin (T2D)"
    ]
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
    figsize = (16, 10)
    palette = "Set2"
    for metric in metrics:
        plt.figure(figsize=figsize)
        sns.boxplot(x="Dataset", y=metric, hue="Method", data=df, palette=palette)

        # Improve readability
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Comparison of {metric} across Datasets")
        plt.legend(title="Method")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Show plot
        plt.show()
        ...


def plot_auroc_kfold(auc_dict, split_type, output_folder, alpha=0.2, n_points=100):
    """
    Plots an aggregated AUROC curve with confidence intervals using k-fold cross-validation results.

    Parameters:
    - y_trues: List of numpy arrays, each containing true labels from a CV fold.
    - y_scores: List of numpy arrays, each containing predicted probabilities for the positive class from the same fold.
    - alpha: Transparency level for the confidence interval (default: 0.2).
    - n_points: Number of points to interpolate the ROC curve for smooth averaging (default: 100).
    """
    rename_dict = {
        'HanniganGD_2017': 'Hannigan (CRC)',
        'FengQ_2015': 'Feng (CRC)',
        'GuptaA_2019': 'Gupta (CRC)',
        'YachidaS_2019': 'Yachida (CRC)',
        'YuJ_2015': 'Yu (CRC)',
        'ThomasAM_2019_c': 'Thomas (CRC)',
        'ZellerG_2014': 'Zeller (CRC)',
        'VogtmannE_2016': 'Vogtmann (CRC)',
        'WirbelJ_2018': 'Wirbel (CRC)',
        'NielsenHB_2014': 'Nielsen (IBD)',
        'HMP_2019_ibdmdb': 'IBDMDB (IBD)',
        'IjazUZ_2017': 'Ijaz (IBD)',
        'KarlssonFH_2013': 'Karlsson (T2D)',
        'QinJ_2012': 'Qin (T2D)'
    }
    all_fprs = np.linspace(0, 1, n_points)  # Fixed set of FPRs for interpolation
    for study in list(auc_dict.keys()):
        plt.figure(figsize=(8, 6))
        for method in list(auc_dict[study].keys()):
            tprs = []
            aucs = []
            y_trues = auc_dict[study][method]['y_true']
            y_scores = auc_dict[study][method]['y_score']
            for y_true, y_score in zip(y_trues, y_scores):
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                tprs.append(np.interp(all_fprs, fpr, tpr))  # Interpolate to standard FPR grid
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = sem(tprs, axis=0)
            plt.plot(all_fprs, mean_tpr, label=f"{method} (Avg. AUC = {np.mean(aucs):.3f})")
            plt.fill_between(all_fprs, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=alpha)

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{rename_dict[study]} ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(f'{output_folder}/{split_type}_roc__{study}.png', dpi=300)
        ...
    ...


def plot_pca(data, meta, disease, test_dataset_name='', n_components=2, distance_metric='braycurtis', figsize=(8, 6), palette="tab10",
             data_type='raw', pca_or_pcoa='pca', save_plot=True):
    rename_dict = {
        0: 'Control',
        1: 'Case'
    }

    def add_ellipses(ax, pca_df, dataset_col, palette, num_std=1):
        """
        Adds confidence ellipses for each dataset in the PCoA plot, matching the scatter plot colors.

        Parameters:
        - ax: Matplotlib axis object.
        - pcoa_df: DataFrame containing "PC1", "PC2", and dataset column.
        - dataset_col: Column indicating dataset groups.
        - palette: Dictionary mapping dataset names to colors.
        """
        datasets = pca_df[dataset_col].unique()

        for dataset in datasets:
            subset = pca_df[pca_df[dataset_col] == dataset]
            x_mean, y_mean = subset["PC1"].mean(), subset["PC2"].mean()
            cov = np.cov(subset[["PC1", "PC2"]].T)
            eigvals, eigvecs = np.linalg.eigh(cov)

            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = (num_std * 2) * np.sqrt(eigvals)  # 1 standard deviation

            # Use the same color as in the scatter plot
            color = palette[dataset]

            ellipse = Ellipse(
                xy=(x_mean, y_mean), width=width, height=height,
                angle=angle, color=color, alpha=.8, fill=False, linewidth=1.5
            )

            ax.add_patch(ellipse)

    meta = meta.replace({"Group": rename_dict})
    # Perform PCA or PCoA
    if pca_or_pcoa == 'pca':
        pca = PCA(n_components=n_components)
        pca_results = pca.fit_transform(data)
        pca_df = pd.DataFrame(pca_results, columns=[f"PC{i + 1}" for i in range(n_components)])

    else:
        distance_matrix = squareform(pdist(data, metric=distance_metric))  # Change metric if needed
        pcoa_results = pcoa(distance_matrix)
        pca_df = pd.DataFrame(pcoa_results.samples.iloc[:, :n_components],
                              columns=[f"PC{i + 1}" for i in range(n_components)])

    # Create a DataFrame for plotting
    pca_df['Dataset'] = meta['Dataset'].values
    pca_df['Group'] = meta['Group'].values

    # Define markers for conditions
    unique_conditions = meta['Group'].unique()
    markers = ["o", "X", "^", "v", "P", "X"][:len(unique_conditions)]  # Extend if needed
    marker_dict = dict(zip(unique_conditions, markers))

    dataset_col = "Dataset"  # Change to your dataset column name
    palette = dict(
        zip(pca_df[dataset_col].unique(), sns.color_palette(palette, n_colors=pca_df[dataset_col].nunique())))

    # Plot PCA with colors for datasets and shapes for conditions
    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        ax=ax,
        x="PC1", y="PC2",
        hue='Dataset', style='Group',
        markers=marker_dict, palette=palette,
        data=pca_df, edgecolor="black", alpha=0.8
    )
    add_ellipses(ax, pca_df, dataset_col, palette, num_std=1)
    if pca_or_pcoa == 'pca':
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
    else:
        plt.xlabel(f"PC1 ({pcoa_results.proportion_explained[0] * 100:.2f}% variance)")
        plt.ylabel(f"PC2 ({pcoa_results.proportion_explained[1] * 100:.2f}% variance)")

    if data_type == 'raw':
        plt.title(f"{pca_or_pcoa.upper()} of raw count data")
    elif data_type == 'ilr':
        plt.title(f"{pca_or_pcoa.upper()} of ILR-transformed data")
    elif data_type == 'clr':
        plt.title(f"{pca_or_pcoa.upper()} of CLR-transformed data")
    else:
        plt.title(f"{pca_or_pcoa.upper()} of MetaDAN embedded data ({test_dataset_name})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Legend")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    if save_plot:
        if data_type == 'raw':
            plt.savefig(f'{output_folder}/{disease}_{pca_or_pcoa}_raw.png', dpi=300)
        elif data_type == 'ilr':
            plt.savefig(f'{output_folder}/{disease}_{pca_or_pcoa}_ilr.png', dpi=300)
        elif data_type == 'clr':
            plt.savefig(f'{output_folder}/{disease}_{pca_or_pcoa}_clr.png', dpi=300)
        else:
            plt.savefig(f'{output_folder}/{disease}_{pca_or_pcoa}_embedding_{test_dataset_name}.png', dpi=300)

    # plt.show()
    plt.close()
    ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model comparisons')
    parser.add_argument('--split_type', type=str, default='loso', help='kfold/loso/toso')
    parser.add_argument('--disease', type=str, default='crc', help='crc/ibd/t2d')
    parser.add_argument('--transform', type=str, default='crc', help='clr/ilr')

    args = parser.parse_args()
    split_type = args.split_type
    disease = args.disease
    transform = args.transform

    output_folder = "./comparison_figures"
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    rename_dict = {
        'HanniganGD_2017': 'Hannigan',
        'FengQ_2015': 'Feng',
        'GuptaA_2019': 'Gupta',
        'YachidaS_2019': 'Yachida',
        'YuJ_2015': 'Yu',
        'ThomasAM_2019_c': 'Thomas',
        'ZellerG_2014': 'Zeller',
        'VogtmannE_2016': 'Vogtmann',
        'WirbelJ_2018': 'Wirbel',
        'NielsenHB_2014': 'Nielsen',
        'HMP_2019_ibdmdb': 'IBDMDB',
        'IjazUZ_2017': 'Ijaz',
        'KarlssonFH_2013': 'Karlsson',
        'QinJ_2012': 'Qin'
    }


    ##############################################################################################
    ################### PCA #######################
    ##############################################################################################
    # for disease in ['crc', 'ibd', 't2d']:
    #     auroc_dict = {}
    #     embedding_dict = {}
    #     results_df = pd.DataFrame(columns=["Method", "Dataset", "Accuracy", "AUC", "F1", "MCC"])
    #     if disease == 'crc':
    #         data, meta = utils.load_CRC_data(rel_abun=False)
    #     elif disease == 'ibd':
    #         data, meta = utils.load_IBD_data(rel_abun=False)
    #     elif disease == 't2d':
    #         data, meta = utils.load_T2D_data(rel_abun=False)
    #     meta = meta.replace({"Dataset": rename_dict})
    #     plot_pca(data, meta, disease, data_type='raw', pca_or_pcoa='pca', figsize=(10, 8))  # raw PCA
    #     plot_pca(data, meta, disease, data_type='raw', pca_or_pcoa='pcoa', figsize=(10, 8))  # raw PCA
    #
    #     if disease == 'crc':
    #         data, meta = utils.load_CRC_data(transform='clr')
    #     elif disease == 'ibd':
    #         data, meta = utils.load_IBD_data(transform='clr')
    #     elif disease == 't2d':
    #         data, meta = utils.load_T2D_data(transform='clr')
    #     meta = meta.replace({"Dataset": rename_dict})
    #     plot_pca(data, meta, disease, data_type='clr', pca_or_pcoa='pca', figsize=(10, 8))  # clr PCA
    #
    #     get_metaDAN_results(disease, 'loso', transform='clr')
    #     embedding_dict = {rename_dict.get(k, k): v for k, v in embedding_dict.items()}
    #     for test_dataset in list(embedding_dict.keys()):
    #         plot_pca(embedding_dict[test_dataset]['embeddings'][-1], meta, disease,
    #                  test_dataset_name=test_dataset, data_type='embedding')

    ##############################################################################################
    ########### k-Fold and LOSO tables ############
    ##############################################################################################

    # for disease in ['crc', 'ibd', 't2d']:
    #     for split_type in ['kfold', 'loso']:
    #         auroc_dict = {}
    #         embedding_dict = {}
    #         results_df = pd.DataFrame(columns=["Method", "Dataset", "Accuracy", "AUC", "F1", "MCC"])
    #         for transform in ['ilr', 'clr']:
    #             get_metaDAN_results(disease, split_type, transform, use_threshold=True)
    #         get_SIAMCAT_results(disease, split_type)
    #         get_metAML_results(disease, split_type)
    #         # if split_type == 'kfold':
    #         #     plot_auroc_kfold(auroc_dict, split_type, output_folder)
    #         t = compute_mean_and_me(results_df, split_type, mean_only=False)
    #         t.to_csv(f'{output_folder}/{split_type}_{disease}_results_table.csv')

    ##############################################################################################
    ############### TOSO heatmaps #################
    ##############################################################################################
    for disease in ['crc', 'ibd', 't2d']:
        results_df = pd.DataFrame(columns=["Method",
                                           "Training Dataset", "Testing Dataset",
                                           "Accuracy", "AUC", "F1", "MCC"])
        auroc_dict = {}
        embedding_dict = {}
        get_metaDAN_results(disease, 'toso', transform='clr')
        get_SIAMCAT_results(disease, 'toso')
        get_metAML_results(disease, 'toso')
        generate_heatmaps(results_df, 'toso', disease, output_folder)

    ...
