import shutil
import subprocess
from functools import partial
import os
import tempfile
from pathlib import Path

import pandas as pd
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
import simulations
import utils

R_path = "/usr/local/bin/Rscript"


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


def get_bandwidth_estimate(model, data):
    """
    :param model:
    :param data:
    :param dataset_ids:
    :param split_type:
    :param subset_proportion:
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


def raytrain_model(config,
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
    net.set_bandwidths(get_bandwidth_estimate(net, train_data))
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
            net.update_bandwidths(get_bandwidth_estimate(net, train_data))

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

            _mkmmd_loss = torch.clamp(target_mkmmd_loss, min=0) * config['target_lambda']

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
                            "auc": roc_auc_score(val_labels[val_classification_idx],
                                                 val_preds.cpu().numpy()),
                            "f1": f1_score(
                                [int(x) for x in val_labels[val_classification_idx]],
                                [0 if nn.functional.sigmoid(val_preds.data[i]) < 0.5 else 1
                                 for i in range(len(val_preds))],
                                average='macro')},
                        checkpoint=checkpoint,
                    )
    return


def raytest_model(best_config,
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

    if (sum(test_labels[test_classification_idx] == 0) > 0) and (sum(test_labels[test_classification_idx] == 1) > 0):
        auc = roc_auc_score(test_labels[test_classification_idx], target_preds)
    else:
        auc = None
    confusion_mat = confusion_matrix(test_labels[test_classification_idx],
                                     [0 if nn.functional.sigmoid(target_preds.data[i]) < 0.5 else 1
                                      for i in range(len(target_preds))])
    return (auc,
            confusion_mat)


def run_raytune(train_data, test_data, val_data,
                num_samples=100, max_num_epochs=100,
                results_folder='ray_results'):
    hyperparams = {
        'batch_size': tune.choice([2 ** i for i in range(3, 5)]),
        'target_lambda': tune.choice([0.5, 1, 2]),
        'learning_rate': tune.loguniform(1e-3, 1e-2),
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
    val_data_ref = ray.put(torch.stack([x[0] for x in val_data]))
    val_labels_ref = ray.put(torch.tensor([x[1] for x in val_data]))
    test_data_ref = ray.put(torch.stack([x[0] for x in test_data]))
    test_labels_ref = ray.put(torch.tensor([x[1] for x in test_data]))

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                partial(raytrain_model,
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

    test_res = raytest_model(best_res.config,
                             best_res,
                             test_data_ref=test_data_ref,
                             test_labels_ref=test_labels_ref
                             )

    print("Best trial test set auc: {}".format(test_res[0]))
    print("Best trial test set confusion matrix")
    print(test_res[1])
    return test_res, best_res.config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mkmmd model hyperparameter tuning')
    parser.add_argument('--disease', type=str, default='crc', help='crc/ibd/t2d')

    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)
    disease = args.disease

    temp_folder = Path('./temp_results').resolve()
    output_folder = Path('./simulation_results').resolve()
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)

    num_trials = 10
    if disease == 'crc':
        test_study = 'GuptaA_2019'
        train_study = 'HanniganGD_2017'
    elif disease == 'ibd':
        ...
    elif disease == 't2d':
        ...
    for alpha in [0.2, 0.4, 0.6, 0.8, 1.0]:
        for proportion_diff in [0.05, 0.1, 0.15, 0.2, 0.25]:
            for effect_size in [2, 4, 6, 8, 10]:
                # run siamcat 10 times with data
                data1, meta1 = simulations.sim_train_test(disease, train_study, test_study, alpha,
                                                          proportion_diff, effect_size)
                data = simulations.convert_data(data1, transform='relative_abundance')
                data.T.to_csv('{}/siamcat_dataset.csv'.format(output_folder))
                meta1.to_csv('{}/siamcat_meta.csv'.format(output_folder))
                cmd = f'{R_path} run_siamcat_sim.R -x \
                                {output_folder}/siamcat_dataset.csv -m {output_folder}/siamcat_meta.csv \
                                -f {train_study} -g {test_study} \
                                -o {output_folder}/ -a {alpha} -p {proportion_diff} -e {effect_size} -d {disease}'
                subprocess.run(cmd, shell=True, executable="/bin/bash")

                # run metaml 10 times with data
                data = simulations.convert_data(data1, transform='relative_abundance')
                data.columns = ['d__' + data.columns[i] for i in range(len(data.columns))]
                data_temp = data.reset_index().rename(columns={'index': 'sample_id'})
                meta_temp = meta1.reset_index().rename(columns={'index': 'sample_id'})
                df_combined = pd.merge(data_temp, meta_temp, on='sample_id', how='inner').T
                df_combined.to_csv(f'{output_folder}/metaml_dataset.csv', index=True, sep='\t', header=False)
                subset_data_cmd = f'python3 ./metaml_code/dataset_selection.py \
                                                    {output_folder}/metaml_dataset.csv \
                                                    {output_folder}/temp_metaml_dataset.txt \
                                                    -z "d__" \
                                                    -s Dataset:train:test \
                                                    -i Dataset:Group'
                classify_cmd = f'python3 ./metaml_code/classification.py \
                                        {output_folder}/temp_metaml_dataset.txt \
                                        {output_folder}/MetAML_{disease}_alpha_{alpha}_proportion_diff_{proportion_diff}_ed_{effect_size}_train_{train_study}_test_{test_study} \
                                        -z "d__" \
                                        -d 1:Group:1 \
                                        -g [] \
                                        -r 10 \
                                        -t Dataset:test'
                full_cmd = f"{subset_data_cmd} && {classify_cmd}"
                subprocess.run(full_cmd, shell=True, executable="/bin/bash")

                # run metadan hparam tuning
                data = simulations.convert_data(data1, transform='clr')
                meta_res = {
                    'test_dataset': test_study,
                    'train_dataset': train_study,
                    'best_hyperparams': None,
                    'best_state_dict': None,
                    'test_aucs': [],
                    'test_confusion_matrices': []
                }
                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder, ignore_errors=True)
                else:
                    os.mkdir(temp_folder)
                train = np.where(meta1['Dataset'] == 'train')[0]
                test = np.where(meta1['Dataset'] == 'test')[0]
                train_idx = meta1.index[train]
                test_idx = meta1.index[test]
                train_temp, validation_temp = train_test_split(np.arange(len(train_idx)),
                                                               test_size=0.2,
                                                               shuffle=True,
                                                               stratify=meta1.loc[train_idx, 'Group'])

                val_idx = train_idx[validation_temp]
                train_idx = train_idx[train_temp]
                train_set, test_set, val_set = load_data_for_training(
                    data, meta1, train_idx, test_idx, val_idx=val_idx)
                raytune_test_res, best_hyperparams = run_raytune(train_set,
                                                                 test_set,
                                                                 val_set,
                                                                 num_samples=250, max_num_epochs=100,
                                                                 results_folder=temp_folder)
                meta_res['best_hyperparams'] = best_hyperparams
                # run metadan 10 times using hparams
                for trial in range(num_trials):
                    train_idx = meta1.index[train]
                    test_idx = meta1.index[test]
                    train_set, test_set = load_data_for_training(
                        data, meta1, train_idx, test_idx)
                    # train model
                    cur_state_dict = utils.train_toso_model(best_hyperparams,
                                                            train_set,
                                                            test_set)
                    # test model
                    cur_auc, cur_conf_mat = utils.test_model(best_hyperparams,
                                                             cur_state_dict,
                                                             test_set)
                    print(cur_auc, cur_conf_mat)
                    # check if best so far and update
                    if trial == 0:
                        meta_res['best_state_dict'] = cur_state_dict
                    else:
                        if cur_auc > np.max(meta_res['test_aucs']):
                            meta_res['best_state_dict'] = cur_state_dict
                    meta_res['test_aucs'].append(cur_auc)
                    meta_res['test_confusion_matrices'].append(cur_conf_mat)

                shutil.rmtree(temp_folder, ignore_errors=True)
                torch.save(meta_res, '{}/MetaDAN_{}_alpha_{}_proportion_diff_{}_ed_{}_train_{}_test_{}.pt'.format(
                    output_folder, disease, alpha, proportion_diff, effect_size, train_study, test_study))
