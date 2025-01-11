import shutil
from functools import partial
import os
import tempfile
from pathlib import Path

import ray

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import random_split, WeightedRandomSampler
from ray import tune
from ray import train as raytrain
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import numpy as np
import argparse

import losses
import networks
import utils


def load_data_for_training(data, meta, train_idx, test_idx, val_idx=None):
    # data is CLR transformed relative abundances, need to standardize for input
    X_train = torch.tensor(data.loc[train_idx].values.astype(float), dtype=torch.float32)
    s, mu = torch.std_mean(X_train, dim=0)
    X_train = (X_train - mu) / s
    X_train = torch.nan_to_num(X_train, nan=0.0)
    label_train = torch.tensor(meta.loc[train_idx, 'Group'], dtype=torch.float32)
    train_set = [[X_train[i], label_train[i]] for i in range(len(train_idx))]

    # test set
    X_test = torch.tensor(data.loc[test_idx].values.astype(float), dtype=torch.float32)
    X_test = (X_test - mu) / s
    X_test = torch.nan_to_num(X_test, nan=0.0)
    label_test = torch.tensor(meta.loc[test_idx, 'Group'], dtype=torch.float32)
    test_set = [[X_test[i], label_test[i]] for i in range(len(test_idx))]

    if val_idx is not None:
        X_val = torch.tensor(data.loc[val_idx].values.astype(float), dtype=torch.float32)
        X_val = (X_val - mu) / s
        X_val = torch.nan_to_num(X_val, nan=0.0)
        label_val = torch.tensor(meta.loc[val_idx, 'Group'], dtype=torch.float32)
        val_set = [[X_val[i], label_val[i]] for i in range(len(val_idx))]
        return train_set, test_set, val_set
    return train_set, test_set


def get_bandwidth_estimate(model, data, dataset_ids, subset_proportion=0.2):
    """

    :param model:
    :param data:
    :param dataset_ids:
    :param subset_proportion:
    :return:
    """
    unique_ids, counts = dataset_ids.unique(return_counts=True)
    samples_per_id = (counts * subset_proportion).long()

    subset_indices = []
    for dataset_id, num_samples in zip(unique_ids, samples_per_id):
        indices = (dataset_ids == dataset_id).nonzero(as_tuple=True)[0]
        sampled_indices = indices[torch.randperm(indices.size(0))[:num_samples]]
        subset_indices.append(sampled_indices)
    subset_indices = torch.tensor(subset_indices)

    embeds, _ = model(data[subset_indices])
    bandwidths = [
        torch.tensor(
            [losses.calc_rbf_bandwidth(embeds[k][dataset_ids[subset_indices] == d]).item()
             for d in dataset_ids[subset_indices].unique()]).mean().item()
        for k in range(len(embeds))
    ]
    return bandwidths


def train_loso_model(config,
                     train_data_ref,
                     train_labels_ref,
                     train_dataset_id_ref,
                     val_data_ref,
                     val_labels_ref,
                     test_data_ref,
                     max_epochs):
    train_data = ray.get(train_data_ref)
    train_labels = ray.get(train_labels_ref)
    train_dataset_id = ray.get(train_dataset_id_ref)
    val_data = ray.get(val_data_ref)
    val_labels = ray.get(val_labels_ref)
    test_data = ray.get(test_data_ref)

    train_classification_idx = torch.nonzero(train_labels <= 1, as_tuple=True)[0]
    train_background_idx = torch.nonzero(train_labels > 1, as_tuple=True)[0]
    val_classification_idx = torch.nonzero(val_labels <= 1, as_tuple=True)[0]

    net = networks.DAN(
        dim=train_data.shape[-1],
        dropout_rate=config['dropout_rate'],
        num_kernels=config['num_kernels'],
        out_dim=1,
        num_hidden_layers=config['num_hidden_layers'],
        embed_size=config['embed_size'],
        num_mmd_layers=config['num_mmd_layers'])

    #  set initial model bandwidths based on subset of training data
    net.set_bandwidths(get_bandwidth_estimate(net, train_data, train_dataset_id))
    mkmmd_loss = losses.MKMMDLoss(num_kernels=config['num_kernels'])
    class_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    start_epoch = 0
    if raytrain.get_checkpoint():
        loaded_checkpoint = raytrain.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state, start_epoch = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt"),
                weights_only=False
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    trainloader_label = torch.utils.data.DataLoader(
        [(train_data[i], train_labels[i], train_dataset_id[i]) for i in train_classification_idx],
        batch_size=int(config["batch_size"]), shuffle=True, drop_last=True
    )

    if len(train_background_idx) > 0:
        trainloader_background = torch.utils.data.DataLoader(
            [(train_data[i], train_labels[i], train_dataset_id[i]) for i in train_background_idx],
            batch_size=int(config["batch_size"]), shuffle=True, drop_last=True
        )

    for epoch in range(start_epoch, max_epochs):
        if epoch % 5 == 0:
            test_buffer = test_data[torch.randperm(len(test_data))[:config["batch_size"] * 2]]
            net.update_bandwidths(get_bandwidth_estimate(net, train_data, train_dataset_id))

        for i, d in enumerate(trainloader_label, 0):
            optimizer.zero_grad()
            source, source_labels, source_dataset_ids = d
            [source_labeled_embeds, source_preds] = net(source)
            if len(train_background_idx) > 0:
                source_background, _ = next(iter(trainloader_background))
                [source_background_embeds, _, source_background_dataset_ids] = net(source_background)
                source_embeds = [
                    torch.cat((t1, t2), dim=0) for t1, t2 in zip(source_labeled_embeds, source_background_embeds)
                ]
                source_ids = torch.hstack([source_dataset_ids, source_background_dataset_ids])
            else:
                source_embeds = source_labeled_embeds
                source_ids = source_dataset_ids

            [target_embeds, _] = net(test_buffer)

            source_label_loss = class_loss(source_preds, source_labels)
            source_mkmmd_loss = torch.tensor(0.0)
            for k in range(config['num_mmd_layers']):
                temp = 0.0
                unique_ids, counts = source_ids.unique(return_counts=True)
                unique_ids = unique_ids[counts > 2]
                if len(unique_ids < 2):
                    break
                num_ids = len(unique_ids)
                num_pairs = 0
                for n in range(num_ids):
                    for m in range(n + 1, num_ids):
                        temp += mkmmd_loss(source_embeds[k][source_ids == unique_ids[n]],
                                           source_embeds[k][source_ids == unique_ids[m]],
                                           net.bandwidth[k],
                                           torch.nn.functional.softmax(net.kernel_weights[k], dim=-1))
                        num_pairs += 1
                source_mkmmd_loss += temp / num_pairs
            target_mkmmd_loss = torch.tensor(0.0)
            for k in range(config['num_mmd_layers']):
                target_mkmmd_loss += mkmmd_loss(source_embeds[k],
                                                target_embeds[k],
                                                net.bandwidth[k],
                                                torch.nn.functional.softmax(net.kernel_weights[k], dim=-1))

            _mkmmd_loss = target_mkmmd_loss * config['target_lambda'] + source_mkmmd_loss * config['source_lambda']

            total_loss = source_label_loss + _mkmmd_loss
            total_loss.backward()
            optimizer.step()

        # Validation metrics
        if epoch % 5 == 0:
            with (torch.no_grad()):
                net.eval()
                [_, val_preds] = net(val_data[val_classification_idx])
                val_label_loss = class_loss(val_preds, val_labels[val_classification_idx])
                net.train()

                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                    torch.save(
                        (net.state_dict(), optimizer.state_dict(), epoch), path
                    )
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                    raytrain.report(
                        {
                            "label_loss": val_label_loss.cpu().numpy(),
                            "accuracy": accuracy_score(
                                [int(x) for x in val_labels[val_classification_idx]],
                                [0 if nn.functional.sigmoid(val_preds.data[i]) < 0.5 else 1
                                 for i in range(len(val_preds))]),
                            "auc": roc_auc_score(val_labels[val_classification_idx],
                                                 val_preds.cpu().numpy()),
                            "f1": f1_score(
                                [int(x) for x in val_labels[val_classification_idx]],
                                [0 if nn.functional.sigmoid(val_preds.data[i]) < 0.5 else 1
                                 for i in range(len(val_preds))],
                                average='macro')},
                        checkpoint=checkpoint,
                    )

    print("Finished Training")
    return
