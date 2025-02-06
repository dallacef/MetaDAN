import shutil
from functools import partial
import os
import tempfile
from pathlib import Path

import ray

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
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
    X_train = torch.tensor(data.loc[train_idx].values.astype(float), dtype=torch.float32)

    s, mu = torch.std_mean(X_train, dim=0)
    X_train = (X_train - mu) / s
    X_train = torch.nan_to_num(X_train, nan=0.0)
    label_train = torch.tensor(meta.loc[train_idx, 'Group'], dtype=torch.float32)
    dataset_train = list(meta.loc[train_idx, 'Dataset'])
    train_set = [[X_train[i], label_train[i], dataset_train[i]] for i in range(len(train_idx))]

    # test set
    X_test = torch.tensor(data.loc[test_idx].values.astype(float), dtype=torch.float32)
    X_test = (X_test - mu) / s
    X_test = torch.nan_to_num(X_test, nan=0.0)
    label_test = torch.tensor(meta.loc[test_idx, 'Group'], dtype=torch.float32)
    dataset_test = list(meta.loc[test_idx, 'Dataset'])
    test_set = [[X_test[i], label_test[i], dataset_test[i]] for i in range(len(test_idx))]

    if val_idx is not None:
        X_val = torch.tensor(data.loc[val_idx].values.astype(float), dtype=torch.float32)
        X_val = (X_val - mu) / s
        X_val = torch.nan_to_num(X_val, nan=0.0)
        label_val = torch.tensor(meta.loc[val_idx, 'Group'], dtype=torch.float32)
        dataset_val = list(meta.loc[val_idx, 'Dataset'])
        val_set = [[X_val[i], label_val[i], dataset_val[i]] for i in range(len(val_idx))]
        return train_set, test_set, val_set
    return train_set, test_set


def get_loso_bandwidth_estimate(model, data, dataset_ids, subset_proportion=0.2):
    """
    :param model:
    :param data:
    :param dataset_ids:
    :param subset_proportion:
    :return:
    """
    dataset_ids = np.array(dataset_ids)
    unique_ids, counts = np.unique(dataset_ids, return_counts=True)
    samples_per_id = (counts * subset_proportion).astype(int)

    subset_indices = []
    for dataset_id, num_samples in zip(unique_ids, samples_per_id):
        indices = (dataset_ids == dataset_id).nonzero()[0]
        sampled_indices = indices[torch.randperm(len(indices))[:num_samples]]
        subset_indices.append(sampled_indices)
    subset_indices = torch.tensor([idx for sublist in subset_indices for idx in sublist]).long()
    embeds, _ = model(data[subset_indices])
    bandwidths = [
        torch.tensor(
            [losses.calc_rbf_bandwidth(embeds[k][dataset_ids[subset_indices] == d]).item()
             for d in np.unique(dataset_ids[subset_indices])]).mean().item()
        for k in range(len(embeds))
    ]
    return bandwidths


def get_toso_bandwidth_estimate(model, data):
    """
    :param model:
    :param data:
    :return:
    """
    num_samples = int(len(data))
    subset_indices = torch.randperm(len(data))[:num_samples]
    embeds, _ = model(data[subset_indices])
    bandwidths = [
        losses.calc_rbf_bandwidth(embeds[k]).item()
        for k in range(len(embeds))
    ]
    return bandwidths


def get_kfold_bandwidth_estimate(model, data):
    """
    :param model:
    :param data:
    :return:
    """
    num_samples = int(len(data))
    subset_indices = torch.randperm(len(data))[:num_samples]
    embeds, _ = model(data[subset_indices])
    bandwidths = [
        losses.calc_rbf_bandwidth(embeds[k]).item()
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
    net.set_bandwidths(get_loso_bandwidth_estimate(net, train_data, train_dataset_id))
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
            net.update_bandwidths(get_loso_bandwidth_estimate(net, train_data, train_dataset_id))

        for i, d in enumerate(trainloader_label, 0):
            optimizer.zero_grad()
            source, source_labels, source_dataset_ids = d
            source_labeled_embeds, source_preds = net(source)
            if len(train_background_idx) > 0:
                source_background, _, source_background_dataset_ids = next(iter(trainloader_background))
                source_background_embeds, _ = net(source_background)
                source_embeds = [
                    torch.cat((t1, t2), dim=0) for t1, t2 in zip(source_labeled_embeds, source_background_embeds)
                ]
                source_ids = np.hstack([source_dataset_ids, source_background_dataset_ids])
            else:
                source_embeds = source_labeled_embeds
                source_ids = np.array(source_dataset_ids)

            target_embeds, _ = net(test_buffer)

            source_label_loss = class_loss(source_preds.flatten(), source_labels)
            source_mkmmd_loss = torch.tensor(0.0)
            for k in range(config['num_mmd_layers']):
                temp = 0.0
                unique_ids, counts = np.unique(source_ids, return_counts=True)
                unique_ids = unique_ids[counts > 2]
                if len(unique_ids) < 2:
                    break
                num_ids = len(unique_ids)
                num_pairs = 0
                for n in range(num_ids - 1):
                    for m in range(n + 1, num_ids):
                        temp += mkmmd_loss(source_embeds[k][source_ids == unique_ids[n]],
                                           source_embeds[k][source_ids == unique_ids[m]],
                                           net.bandwidths[k],
                                           torch.nn.functional.softmax(net.kernel_weights[k], dim=-1))
                        num_pairs += 1
                source_mkmmd_loss += temp / num_pairs
            target_mkmmd_loss = torch.tensor(0.0)
            for k in range(config['num_mmd_layers']):
                target_mkmmd_loss += mkmmd_loss(source_embeds[k],
                                                target_embeds[k],
                                                net.bandwidths[k],
                                                torch.nn.functional.softmax(net.kernel_weights[k], dim=-1))

            _mkmmd_loss = target_mkmmd_loss * config['target_lambda'] + source_mkmmd_loss * config['source_lambda']

            total_loss = source_label_loss + _mkmmd_loss
            total_loss.backward()
            optimizer.step()

        # Validation metrics
        if epoch % 5 == 0:
            with (torch.no_grad()):
                net.eval()
                _, val_preds = net(val_data[val_classification_idx])
                val_label_loss = class_loss(val_preds.flatten(), val_labels[val_classification_idx])
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


def train_toso_model(config,
                     train_data_ref,
                     train_labels_ref,
                     val_data_ref,
                     val_labels_ref,
                     test_data_ref,
                     max_epochs):
    train_data = ray.get(train_data_ref)
    train_labels = ray.get(train_labels_ref)
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

    #  set initial model bandwidths based on  training data
    net.set_bandwidths(get_toso_bandwidth_estimate(net, train_data))
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
        [(train_data[i], train_labels[i]) for i in train_classification_idx],
        batch_size=int(config["batch_size"]), shuffle=True, drop_last=True
    )

    if len(train_background_idx) > 0:
        trainloader_background = torch.utils.data.DataLoader(
            [(train_data[i], train_labels[i]) for i in train_background_idx],
            batch_size=int(config["batch_size"]), shuffle=True, drop_last=True
        )

    for epoch in range(start_epoch, max_epochs):
        if epoch % 5 == 0:
            test_buffer = test_data[torch.randperm(len(test_data))[:config["batch_size"] * 2]]
            net.update_bandwidths(get_toso_bandwidth_estimate(net, train_data))

        for i, d in enumerate(trainloader_label, 0):
            optimizer.zero_grad()
            source, source_labels = d
            source_labeled_embeds, source_preds = net(source)
            if len(train_background_idx) > 0:
                source_background, _ = next(iter(trainloader_background))
                source_background_embeds, _ = net(source_background)
                source_embeds = [
                    torch.cat((t1, t2), dim=0) for t1, t2 in zip(source_labeled_embeds, source_background_embeds)
                ]
            else:
                source_embeds = source_labeled_embeds
            target_embeds, _ = net(test_buffer)

            source_label_loss = class_loss(source_preds.flatten(), source_labels)
            target_mkmmd_loss = torch.tensor(0.0)
            for k in range(config['num_mmd_layers']):
                target_mkmmd_loss += mkmmd_loss(source_embeds[k],
                                                target_embeds[k],
                                                net.bandwidths[k],
                                                torch.nn.functional.softmax(net.kernel_weights[k], dim=-1))

            _mkmmd_loss = target_mkmmd_loss * config['target_lambda']

            total_loss = source_label_loss + _mkmmd_loss
            total_loss.backward()
            optimizer.step()

        # Validation metrics
        if epoch % 5 == 0:
            with (torch.no_grad()):
                net.eval()
                [_, val_preds] = net(val_data[val_classification_idx])
                val_label_loss = class_loss(val_preds.flatten(), val_labels[val_classification_idx])
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


def train_kfold_model(config,
                      train_data_ref,
                      train_labels_ref,
                      val_data_ref,
                      val_labels_ref,
                      test_data_ref,
                      max_epochs):
    train_data = ray.get(train_data_ref)
    train_labels = ray.get(train_labels_ref)
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

    #  set initial model bandwidths based on  training data
    net.set_bandwidths(get_kfold_bandwidth_estimate(net, train_data))
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
        [(train_data[i], train_labels[i]) for i in train_classification_idx],
        batch_size=int(config["batch_size"]), shuffle=True, drop_last=True
    )

    if len(train_background_idx) > 0:
        trainloader_background = torch.utils.data.DataLoader(
            [(train_data[i], train_labels[i]) for i in train_background_idx],
            batch_size=int(config["batch_size"]), shuffle=True, drop_last=True
        )

    for epoch in range(start_epoch, max_epochs):
        if epoch % 5 == 0:
            net.update_bandwidths(get_kfold_bandwidth_estimate(net, train_data))

        for i, d in enumerate(trainloader_label, 0):
            optimizer.zero_grad()
            source, source_labels = d
            source_labeled_embeds, source_preds = net(source)
            if len(train_background_idx) > 0:
                source_background, _ = next(iter(trainloader_background))
                source_background_embeds, _ = net(source_background)
                source_embeds = [
                    torch.cat((t1, t2), dim=0) for t1, t2 in zip(source_labeled_embeds, source_background_embeds)
                ]
            else:
                source_embeds = source_labeled_embeds
            target_embeds, _ = net(test_data)

            source_label_loss = class_loss(source_preds.flatten(), source_labels)
            target_mkmmd_loss = torch.tensor(0.0)
            for k in range(config['num_mmd_layers']):
                target_mkmmd_loss += mkmmd_loss(source_embeds[k],
                                                target_embeds[k],
                                                net.bandwidths[k],
                                                torch.nn.functional.softmax(net.kernel_weights[k], dim=-1))

            _mkmmd_loss = target_mkmmd_loss * config['target_lambda']

            total_loss = source_label_loss + _mkmmd_loss
            total_loss.backward()
            optimizer.step()

        # Validation metrics
        if epoch % 5 == 0:
            with (torch.no_grad()):
                net.eval()
                [_, val_preds] = net(val_data[val_classification_idx])
                val_label_loss = class_loss(val_preds.flatten(), val_labels[val_classification_idx])
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


def test_stats(best_config,
               best_result,
               test_data_ref,
               test_labels_ref):
    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    best_model_state, _, __ = torch.load(checkpoint_path, weights_only=False)

    test_data = ray.get(test_data_ref)
    test_labels = ray.get(test_labels_ref)

    test_classification_idx = torch.nonzero(test_labels <= 1, as_tuple=True)[0]

    net = networks.DAN(
        dim=test_data.shape[-1],
        dropout_rate=best_config['dropout_rate'],
        num_kernels=best_config['num_kernels'],
        out_dim=1,
        num_hidden_layers=best_config['num_hidden_layers'],
        embed_size=best_config['embed_size'],
        num_mmd_layers=best_config['num_mmd_layers'])

    net.load_state_dict(best_model_state)
    with torch.no_grad():
        net.eval()
        [_, preds] = net(test_data)
        target_preds = preds[test_classification_idx].flatten()
        net.train()

    acc = accuracy_score([int(x) for x in test_labels[test_classification_idx]],
                         [0 if nn.functional.sigmoid(target_preds.data[i]) < 0.5 else 1
                          for i in range(len(target_preds))])
    if (sum(test_labels[test_classification_idx] == 0) > 0) and (sum(test_labels[test_classification_idx] == 1) > 0):
        f1 = f1_score(test_labels[test_classification_idx],
                      [0 if nn.functional.sigmoid(target_preds.data[i]) < 0.5 else 1
                       for i in range(len(target_preds))],
                      average='macro')
        auc = roc_auc_score(test_labels[test_classification_idx], target_preds)
        confusion_mat = confusion_matrix(test_labels[test_classification_idx],
                                         [0 if nn.functional.sigmoid(target_preds.data[i]) < 0.5 else 1
                                          for i in range(len(target_preds))])
    else:
        f1 = None
        auc = None
        confusion_mat = None
    return (acc,
            f1,
            auc,
            confusion_mat,
            best_model_state)


def run_raytune(train_data, test_data, val_data,
                num_samples=100, max_num_epochs=100,
                results_folder='ray_results', split_type='loso'):
    if split_type == 'kfold':
        hyperparams = {
            'batch_size': tune.choice([2 ** i for i in range(3, 5)]),
            'target_lambda': tune.choice([0.5, 1, 2]),
            'learning_rate': tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.choice([0.01, 0.05, 0.1]),
            'num_kernels': tune.choice([1, 2, 3]),
            'dropout_rate': tune.choice([0.1, 0.3]),
            'num_hidden_layers': tune.choice([1, 2, 3]),
            'embed_size': tune.choice([16, 32, 64]),
            'num_mmd_layers': tune.choice([1, 2]),
            'out_dim': 1
        }

    elif split_type == 'loso':
        hyperparams = {
            'batch_size': tune.choice([2 ** i for i in range(4, 6)]),
            'source_lambda': tune.choice([1, 2.5, 3]),
            'target_lambda': tune.choice([1, 2.5, 3]),
            'learning_rate': tune.loguniform(1e-3, 1e-1),
            "weight_decay": tune.choice([0.01, 0.05, 0.1]),
            'num_kernels': tune.choice([3, 4, 5]),
            'dropout_rate': tune.choice([0.1, 0.3]),
            'num_hidden_layers': tune.choice([1, 2, 3]),
            'embed_size': tune.choice([32, 64]),
            'num_mmd_layers': tune.choice([2, 3]),
            'out_dim': 1
        }

    elif split_type == 'toso':
        hyperparams = {
            'batch_size': tune.choice([2 ** i for i in range(3, 5)]),
            'target_lambda': tune.choice([0.5, 1, 2]),
            'learning_rate': tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.choice([0.01, 0.05, 0.1]),
            'num_kernels': tune.choice([3, 4, 5]),
            'dropout_rate': tune.choice([0.1, 0.3]),
            'num_hidden_layers': tune.choice([1, 2, 3]),
            'embed_size': tune.choice([32, 64]),
            'num_mmd_layers': tune.choice([1, 2, 3]),
            'out_dim': 1
        }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=2,
    )
    train_data_ref = ray.put(torch.stack([x[0] for x in train_data]))
    train_labels_ref = ray.put(torch.tensor([x[1] for x in train_data]))
    train_dataset_id_ref = ray.put([x[2] for x in train_data])
    val_data_ref = ray.put(torch.stack([x[0] for x in val_data]))
    val_labels_ref = ray.put(torch.tensor([x[1] for x in val_data]))
    test_data_ref = ray.put(torch.stack([x[0] for x in test_data]))
    test_labels_ref = ray.put(torch.tensor([x[1] for x in test_data]))

    if split_type == 'loso':
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    partial(train_loso_model,
                            train_data_ref=train_data_ref, train_labels_ref=train_labels_ref,
                            train_dataset_id_ref=train_dataset_id_ref,
                            val_data_ref=val_data_ref, val_labels_ref=val_labels_ref,
                            test_data_ref=test_data_ref,
                            max_epochs=max_num_epochs
                            )),
                resources={"cpu": 1}
            ),
            tune_config=tune.TuneConfig(
                metric="auc",
                mode="max",
                scheduler=scheduler,
                num_samples=num_samples,
            ),
            run_config=raytrain.RunConfig(storage_path=results_folder, verbose=0),
            param_space=hyperparams,
        )
    elif split_type == 'toso':
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    partial(train_toso_model,
                            train_data_ref=train_data_ref, train_labels_ref=train_labels_ref,
                            val_data_ref=val_data_ref, val_labels_ref=val_labels_ref,
                            test_data_ref=test_data_ref,
                            max_epochs=max_num_epochs
                            )),
                resources={"cpu": 1}
            ),
            tune_config=tune.TuneConfig(
                metric="auc",
                mode="max",
                scheduler=scheduler,
                num_samples=num_samples,
            ),
            run_config=raytrain.RunConfig(storage_path=results_folder, verbose=0),
            param_space=hyperparams,
        )
    else:
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    partial(train_kfold_model,
                            train_data_ref=train_data_ref, train_labels_ref=train_labels_ref,
                            val_data_ref=val_data_ref, val_labels_ref=val_labels_ref,
                            test_data_ref=test_data_ref,
                            max_epochs=max_num_epochs
                            )),
                resources={"cpu": 1}
            ),
            tune_config=tune.TuneConfig(
                metric="auc",
                mode="max",
                scheduler=scheduler,
                num_samples=num_samples,
            ),
            run_config=raytrain.RunConfig(storage_path=results_folder, verbose=0),
            param_space=hyperparams,
        )
    result = tuner.fit()
    best_res = result.get_best_result("auc", "max", "last-10-avg")
    print("Best trial config: {}".format(best_res.config))
    print(f"Best trial final validation auc: {best_res.metrics['auc']}")
    print(f"Best trial final validation f1: {best_res.metrics['f1']}")
    print(f"Best trial final validation label loss: {best_res.metrics['label_loss']}")
    print(f"Best trial final validation accuracy: {best_res.metrics['accuracy']}")

    test_res = test_stats(best_res.config,
                          best_res,
                          test_data_ref=test_data_ref,
                          test_labels_ref=test_labels_ref
                          )

    print("Best trial test set auc: {}".format(test_res[2]))
    print("Best trial test set f1: {}".format(test_res[1]))
    print("Best trial test set accuracy: {}".format(test_res[0]))
    print("Best trial test set confusion matrix")
    print(test_res[3])
    return test_res, best_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mkmmd model hyperparameter tuning')
    parser.add_argument('--split_type', type=str, default='loso', help='kfold/loso/toso')
    parser.add_argument('--disease', type=str, default='crc', help='crc/ibd/t2d')
    parser.add_argument('--transform', type=str, default='relative_abundance',
                        help='relative_abundance/clr/ilr')

    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    split_type = args.split_type
    disease = args.disease
    transform = args.transform

    temp_folder = Path('./temp_results').resolve()
    output_folder = Path('./dan_results').resolve()
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    #####################################
    # LOSO
    #####################################
    if split_type == 'loso':
        num_trials = 10
        if disease == 'crc':
            data, meta = utils.load_CRC_data(transform=transform)
        elif disease == 'ibd':
            data, meta = utils.load_IBD_data(transform=transform)
        elif disease == 't2d':
            data, meta = utils.load_T2D_data(transform=transform)

        datasets = meta['Dataset'].unique()
        for dset in datasets:
            res = []
            for trial in range(num_trials):
                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder, ignore_errors=True)
                else:
                    os.mkdir(temp_folder)

                train = np.where(meta['Dataset'] != dset)[0]
                test = np.where(meta['Dataset'] == dset)[0]
                train_idx = meta.index[train]
                test_idx = meta.index[test]
                train_temp, validation_temp = train_test_split(np.arange(len(train_idx)),
                                                               test_size=0.2,
                                                               shuffle=True,
                                                               stratify=meta.loc[train_idx, 'Group'])

                val_idx = train_idx[validation_temp]
                train_idx = train_idx[train_temp]
                train_set, test_set, val_set = load_data_for_training(
                    data, meta, train_idx, test_idx, val_idx=val_idx)

                tune_res = run_raytune(train_set,
                                       test_set,
                                       val_set,
                                       num_samples=100, max_num_epochs=100,
                                       results_folder=temp_folder, split_type=split_type)
                res.append({
                    'test_dataset': dset,
                    'train_idx': train_idx,
                    'test_idx': test_idx,
                    'val_idx': val_idx,
                    'test_acc': tune_res[0][0],
                    'test_f1': tune_res[0][1],
                    'test_auc': tune_res[0][2],
                    'test_confusion_matrix': tune_res[0][3],
                    'best_state_dict': tune_res[0][4],
                    'best_params': tune_res[1].config
                })
            shutil.rmtree(temp_folder, ignore_errors=True)

            torch.save(res, '{}/{}_{}_{}_{}.pt'.format(output_folder, split_type, disease, dset, transform))

    #####################################
    # TOSO
    #####################################
    elif split_type == 'toso':
        num_trials = 10
        if disease == 'crc':
            datasets = ['FengQ_2015', 'GuptaA_2019', 'HanniganGD_2017', 'ThomasAM_2019_c',
                        'VogtmannE_2016', 'WirbelJ_2018', 'YachidaS_2019', 'YuJ_2015', 'ZellerG_2014']
        elif disease == 'ibd':
            datasets = ['HMP_2019_ibdmdb', 'IjazUZ_2017', 'NielsenHB_2014']
        elif disease == 't2d':
            datasets = ['KarlssonFH_2013', 'QinJ_2012']

        for dset in datasets:
            for dset2 in datasets:
                if dset == dset2:
                    continue
                if disease == 'crc':
                    data, meta = utils.load_CRC_data(studies_to_include=[dset, dset2], transform=transform)
                elif disease == 'ibd':
                    data, meta = utils.load_IBD_data(studies_to_include=[dset, dset2], transform=transform)
                elif disease == 't2d':
                    data, meta = utils.load_T2D_data(studies_to_include=[dset, dset2], transform=transform)

                res = []
                for trial in range(num_trials):
                    if os.path.exists(temp_folder):
                        shutil.rmtree(temp_folder, ignore_errors=True)
                    else:
                        os.mkdir(temp_folder)

                    train = np.where(meta['Dataset'] == dset)[0]
                    test = np.where(meta['Dataset'] == dset2)[0]
                    train_idx = meta.index[train]
                    test_idx = meta.index[test]
                    train_temp, validation_temp = train_test_split(np.arange(len(train_idx)),
                                                                   test_size=0.2,
                                                                   shuffle=True,
                                                                   stratify=meta.loc[train_idx, 'Group'])

                    val_idx = train_idx[validation_temp]
                    train_idx = train_idx[train_temp]
                    train_set, test_set, val_set = load_data_for_training(
                        data, meta, train_idx, test_idx, val_idx=val_idx)
                    tune_res = run_raytune(train_set,
                                           test_set,
                                           val_set,
                                           num_samples=100, max_num_epochs=100,
                                           results_folder=temp_folder, split_type=split_type)
                    res.append({
                        'train_dataset': dset,
                        'test_dataset': dset2,
                        'train_idx': train_idx,
                        'test_idx': test_idx,
                        'val_idx': val_idx,
                        'test_acc': tune_res[0][0],
                        'test_f1': tune_res[0][1],
                        'test_auc': tune_res[0][2],
                        'test_confusion_matrix': tune_res[0][3],
                        'best_state_dict': tune_res[0][4],
                        'best_params': tune_res[1].config

                    })
                shutil.rmtree(temp_folder, ignore_errors=True)
                torch.save(res, '{}/{}_{}_{}_train_{}_test_{}.pt'.format(output_folder, split_type, disease, dset,
                                                                         dset2, transform))

    #####################################
    # KFOLD
    #####################################
    elif split_type == 'kfold':
        num_trials = 5
        num_folds = 5

        if disease == 'crc':
            # datasets = ['feng', 'hannigan', 'thomas', 'vogtmann', 'yu', 'zeller']
            datasets = ['FengQ_2015', 'GuptaA_2019', 'HanniganGD_2017', 'ThomasAM_2019_c', 'VogtmannE_2016',
                        'WirbelJ_2018', 'YachidaS_2019', 'YuJ_2015', 'ZellerG_2014']
        elif disease == 'ibd':
            datasets = ['HMP_2019_ibdmdb', 'IjazUZ_2017', 'NielsenHB_2014']
        elif disease == 't2d':
            datasets = ['KarlssonFH_2013', 'QinJ_2012']

        for dataset in datasets:
            if dataset != 'ThomasAM_2019_c':
                continue
            if disease == 'crc':
                # data, meta = utils.load_CRC_data(studies=[dataset])
                data, meta = utils.load_CRC_data(studies_to_include=[dataset], transform=transform)
            elif disease == 'ibd':
                data, meta = utils.load_IBD_data(studies_to_include=[dataset], transform=transform)
            elif disease == 't2d':
                data, meta = utils.load_T2D_data(studies_to_include=[dataset], transform=transform)

            res = []
            for trial in range(num_trials):
                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder, ignore_errors=True)
                else:
                    os.mkdir(temp_folder)

                kf = StratifiedKFold(n_splits=num_folds, shuffle=True)
                splits = kf.split(np.zeros(len(meta.loc[meta['Dataset'] == dataset])),
                                  list(meta.loc[meta['Dataset'] == dataset, 'Group']))
                for i, (train, test) in enumerate(splits):
                    train_idx = meta.loc[meta['Dataset'] == dataset].index[train]
                    test_idx = meta.loc[meta['Dataset'] == dataset].index[test]
                    train_temp, validation_temp = train_test_split(
                        np.arange(len(train_idx)),
                        test_size=0.2,
                        shuffle=True,
                        stratify=meta.loc[train_idx, 'Group'])
                    val_idx = train_idx[validation_temp]
                    train_idx = train_idx[train_temp]
                    train_set, test_set, val_set = load_data_for_training(
                        data, meta, train_idx, test_idx, val_idx=val_idx)
                    tune_res = run_raytune(train_set,
                                           test_set,
                                           val_set,
                                           num_samples=100, max_num_epochs=100,
                                           results_folder=temp_folder, split_type=split_type)
                    res.append({
                        'test_dataset': dataset,
                        'train_idx': train_idx,
                        'test_idx': test_idx,
                        'val_idx': val_idx,
                        'test_acc': tune_res[0][0],
                        'test_f1': tune_res[0][1],
                        'test_auc': tune_res[0][2],
                        'test_confusion_matrix': tune_res[0][3],
                        'best_state_dict': tune_res[0][4],
                        'best_params': tune_res[1].config

                    })
                shutil.rmtree(temp_folder, ignore_errors=True)

            torch.save(res, '{}/{}_{}_{}_{}.pt'.format(output_folder, split_type, disease, dataset, transform))
