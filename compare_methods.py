import os
import re
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, confusion_matrix, auc
import networks
import utils


def get_metaDAN_results(disease, split_type):
    if split_type == 'kfold':
        pattern = rf"^{split_type}_{disease}_[a-zA-Z0-9]+\.pt$"
    elif split_type == 'loso':
        pattern = rf"^{split_type}_{disease}_[a-zA-Z0-9]+\.pt$"
    elif split_type == 'toso':
        pattern = rf"^{split_type}_[a-zA-Z0-9]+_train_[a-zA-Z0-9]+_test\.pt$"

    for filename in os.listdir('dan_results'):
        if re.match(pattern, filename):
            filepath = os.path.join('dan_results', filename)
            ray_res = torch.load(filepath, weights_only=False)
            for i in range(len(ray_res)):
                save_results(
                    'MetaDAN',
                    ray_res[i]['test_dataset'],
                    metric_from_confusion_matrix(ray_res[i]['confusion_matrix'], metric='acc'),
                    ray_res[i]['test_auc'],
                    metric_from_confusion_matrix(ray_res[i]['confusion_matrix'], metric='f1'),
                    metric_from_confusion_matrix(ray_res[i]['confusion_matrix'], metric='mcc'),
                    metric_from_confusion_matrix(ray_res[i]['confusion_matrix'], metric='tpr'),
                    metric_from_confusion_matrix(ray_res[i]['confusion_matrix'], metric='fpr')
                )


def get_SIAMCAT_results(disease, split_type):
    ...


def get_metAML_results(disease, split_type):
    ...


