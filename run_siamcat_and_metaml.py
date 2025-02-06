import argparse
import os
import subprocess

import numpy as np
import pandas as pd

import utils

R_path = "/usr/local/bin/Rscript"


def run_siamcat(disease, split_type, output_folder="SIAMCAT_results"):
    """
    SIAMCAT (Wirbel et al. 2021)
    * input is relative abundance data
    """
    # todo fix data loading so that testing data isn't used in OTU keep calcualtions
    if os.path.isdir(output_folder):
        ...
    else:
        os.mkdir(output_folder)

    if disease == 'crc':
        data, meta = utils.load_CRC_data()
    elif disease == 'ibd':
        data, meta = utils.load_IBD_data()
    elif disease == 't2d':
        data, meta = utils.load_T2D_data()

    data.T.to_csv('./{}/dataset.csv'.format(output_folder))
    meta.to_csv('./{}/meta.csv'.format(output_folder))
    cmd = f'{R_path} run_siamcat.R -x \
    ./{output_folder}/dataset.csv -m ./{output_folder}/meta.csv \
    -o {output_folder}/ -s {split_type} -d {disease}'
    # todo remove tax argument from SIAMCAT R file

    subprocess.run(cmd, shell=True, executable="/bin/bash")


def run_metaml(disease, split_type, output_folder="MetAML_results"):
    # todo fix data loading so that testing data isn't used in OTU keep calcualtions
    """
    MetAML (Pasolli et al. 20216)
    * input is relative abundance data
    """
    if os.path.isdir(output_folder):
        ...
    else:
        os.mkdir(output_folder)

    if disease == 'crc':
        data, meta = utils.load_CRC_data()
    elif disease == 'ibd':
        data, meta = utils.load_IBD_data()
    elif disease == 't2d':
        data, meta = utils.load_T2D_data()
    data.columns = ['d__' + data.columns[i] for i in range(len(data.columns))]
    data_temp = data.reset_index().rename(columns={'index': 'sample_id'})
    meta_temp = meta.reset_index().rename(columns={'index': 'sample_id'})
    df_combined = pd.merge(data_temp, meta_temp, on='sample_id', how='inner').T
    df_combined.to_csv(f'./{output_folder}/dataset.csv', index=True, sep='\t', header=False)

    for d in meta['Dataset'].unique():
        if split_type == 'loso':
            subset_data_cmd = f'python3 ./metaml_code/dataset_selection.py \
                                            ./{output_folder}/dataset.csv \
                                            ./{output_folder}/temp_dataset.txt \
                                            -z "d__" \
                                            -i Dataset:Group'
            classify_cmd = f'python3 ./metaml_code/classification.py \
                            ./{output_folder}/temp_dataset.txt \
                            ./{output_folder}/{split_type}_{disease}_{d} \
                            -z "d__" \
                            -d 1:Group:1 \
                            -g [] \
                            -r 10 \
                            -t Dataset:{d}'
            full_cmd = f"{subset_data_cmd} && {classify_cmd}"
            subprocess.run(full_cmd, shell=True, executable="/bin/bash")

        elif split_type == 'toso':
            for d2 in meta['Dataset'].unique():
                if d == d2:
                    continue

                subset_data_cmd = f'python3 ./metaml_code/dataset_selection.py \
                                    ./metaml_results/dataset.csv \
                                    ./metaml_results/temp_dataset.txt \
                                    -z "d__" \
                                    -s Dataset:{d}:{d2} \
                                    -i Dataset:Group'
                classify_cmd = f'python3 ./metaml_code/classification.py \
                                            ./{output_folder}/temp_dataset.txt \
                                            ./{output_folder}/{split_type}_{disease}_{d}_train_{d2}_test \
                                            -z "d__" \
                                            -d 1:Group:1 \
                                            -g [] \
                                            -r 10 \
                                            -t Dataset:{d2}'
                full_cmd = f"{subset_data_cmd} && {classify_cmd}"
                subprocess.run(full_cmd, shell=True, executable="/bin/bash")

        elif split_type == 'kfold':
            subset_data_cmd = f'python3 ./metaml_code/dataset_selection.py \
                                            ./{output_folder}/dataset.csv \
                                            ./{output_folder}/temp_dataset.txt \
                                            -z "d__" \
                                            -s Dataset:{d} \
                                            -i Group'
            classify_cmd = f'python3 ./metaml_code/classification.py \
                            ./{output_folder}/temp_dataset.txt \
                            ./{output_folder}/{split_type}_{disease}_{d} \
                            -z "d__" \
                            -d 1:Group:1 \
                            -g [] \
                            -r 5 \
                            -p 5'
            full_cmd = f"{subset_data_cmd} && {classify_cmd}"
            subprocess.run(full_cmd, shell=True, executable="/bin/bash")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get results for SIAMCAT and MetAML')
    parser.add_argument('--split_type', type=str, default='kfold', help='kfold/loso/toso')
    parser.add_argument('--disease', type=str, default='crc', help='crc/ibd/t2d')

    args = parser.parse_args()
    split_type = args.split_type
    disease = args.disease

    run_siamcat(disease, split_type)
    run_metaml(disease, split_type)