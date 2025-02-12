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
    if os.path.isdir(output_folder):
        ...
    else:
        os.mkdir(output_folder)

    if disease == 'crc':
        _, meta1 = utils.load_CRC_data()
    elif disease == 'ibd':
        _, meta1 = utils.load_IBD_data()
    elif disease == 't2d':
        _, meta1 = utils.load_T2D_data()

    if split_type == 'loso':
        for dset in meta1['Dataset'].unique():
            if disease == 'crc':
                data, meta = utils.load_CRC_data(num_feat=150,
                                                 train_studies=list(
                                                     meta1.loc[meta1['Dataset'] != dset]['Dataset'].unique()))
            elif disease == 'ibd':
                data, meta = utils.load_IBD_data(num_feat=150,
                                                 train_studies=list(
                                                     meta1.loc[meta1['Dataset'] != dset]['Dataset'].unique()))
            elif disease == 't2d':
                data, meta = utils.load_T2D_data(num_feat=150,
                                                 train_studies=list(
                                                     meta1.loc[meta1['Dataset'] != dset]['Dataset'].unique()))

            data.T.to_csv('./{}/dataset.csv'.format(output_folder))
            meta.to_csv('./{}/meta.csv'.format(output_folder))
            cmd = f'{R_path} run_siamcat_loso.R -x \
            ./{output_folder}/dataset.csv -m ./{output_folder}/meta.csv \
            -o {output_folder}/ -n {dset} -d {disease}'
            subprocess.run(cmd, shell=True, executable="/bin/bash")

    elif split_type == 'toso':
        for dset in meta1['Dataset'].unique():
            for dset2 in meta1['Dataset'].unique():
                if dset == dset2:
                    continue
                if disease == 'crc':
                    data, meta = utils.load_CRC_data(studies_to_include=[dset, dset2],
                                                     num_feat=150, train_studies=[dset])
                elif disease == 'ibd':
                    data, meta = utils.load_IBD_data(studies_to_include=[dset, dset2],
                                                     num_feat=150, train_studies=[dset])
                elif disease == 't2d':
                    data, meta = utils.load_T2D_data(studies_to_include=[dset, dset2],
                                                     num_feat=150, train_studies=[dset])

                data.T.to_csv('./{}/dataset.csv'.format(output_folder))
                meta.to_csv('./{}/meta.csv'.format(output_folder))
                cmd = f'{R_path} run_siamcat_toso.R -x \
                ./{output_folder}/dataset.csv -m ./{output_folder}/meta.csv \
                -o {output_folder}/ -a {dset} -b {dset2} -d {disease}'
                subprocess.run(cmd, shell=True, executable="/bin/bash")

    elif split_type == 'kfold':
        for dset in meta1['Dataset'].unique():
            if disease == 'crc':
                data, meta = utils.load_CRC_data(studies_to_include=[dset],
                                                 num_feat=150)
            elif disease == 'ibd':
                data, meta = utils.load_IBD_data(studies_to_include=[dset],
                                                 num_feat=150)
            elif disease == 't2d':
                data, meta = utils.load_T2D_data(studies_to_include=[dset],
                                                 num_feat=150)

            data.T.to_csv('./{}/dataset.csv'.format(output_folder))
            meta.to_csv('./{}/meta.csv'.format(output_folder))
            cmd = f'{R_path} run_siamcat_kfold.R -x \
            ./{output_folder}/dataset.csv -m ./{output_folder}/meta.csv \
            -o {output_folder}/ -n {dset} -d {disease}'
            subprocess.run(cmd, shell=True, executable="/bin/bash")


def run_metaml(disease, split_type, output_folder="MetAML_results"):
    """
    MetAML (Pasolli et al. 20216)
    * input is relative abundance data
    """
    if os.path.isdir(output_folder):
        ...
    else:
        os.mkdir(output_folder)

    if disease == 'crc':
        _, meta1 = utils.load_CRC_data()
    elif disease == 'ibd':
        _, meta1 = utils.load_IBD_data()
    elif disease == 't2d':
        _, meta1 = utils.load_T2D_data()

    if split_type == 'loso':
        for dset in meta1['Dataset'].unique():
            if disease == 'crc':
                data, meta = utils.load_CRC_data(num_feat=150,
                                                 train_studies=list(
                                                     meta1.loc[meta1['Dataset'] != dset]['Dataset'].unique()))
            elif disease == 'ibd':
                data, meta = utils.load_IBD_data(num_feat=150,
                                                 train_studies=list(
                                                     meta1.loc[meta1['Dataset'] != dset]['Dataset'].unique()))
            elif disease == 't2d':
                data, meta = utils.load_T2D_data(num_feat=150,
                                                 train_studies=list(
                                                     meta1.loc[meta1['Dataset'] != dset]['Dataset'].unique()))

            data.columns = ['d__' + data.columns[i] for i in range(len(data.columns))]
            data_temp = data.reset_index().rename(columns={'index': 'sample_id'})
            meta_temp = meta.reset_index().rename(columns={'index': 'sample_id'})
            df_combined = pd.merge(data_temp, meta_temp, on='sample_id', how='inner').T
            df_combined.to_csv(f'./{output_folder}/dataset.csv', index=True, sep='\t', header=False)

            subset_data_cmd = f'python3 ./metaml_code/dataset_selection.py \
                                            ./{output_folder}/dataset.csv \
                                            ./{output_folder}/temp_dataset.txt \
                                            -z "d__" \
                                            -i Dataset:Group'
            classify_cmd = f'python3 ./metaml_code/classification.py \
                            ./{output_folder}/temp_dataset.txt \
                            ./{output_folder}/{split_type}_{disease}_{dset} \
                            -z "d__" \
                            -d 1:Group:1 \
                            -g [] \
                            -r 10 \
                            -t Dataset:{dset}'
            full_cmd = f"{subset_data_cmd} && {classify_cmd}"
            subprocess.run(full_cmd, shell=True, executable="/bin/bash")

    elif split_type == 'toso':
        for dset in meta1['Dataset'].unique():
            for dset2 in meta1['Dataset'].unique():
                if dset == dset2:
                    continue
                if disease == 'crc':
                    data, meta = utils.load_CRC_data(studies_to_include=[dset, dset2],
                                                     num_feat=150, train_studies=[dset])
                elif disease == 'ibd':
                    data, meta = utils.load_IBD_data(studies_to_include=[dset, dset2],
                                                     num_feat=150, train_studies=[dset])
                elif disease == 't2d':
                    data, meta = utils.load_T2D_data(studies_to_include=[dset, dset2],
                                                     num_feat=150, train_studies=[dset])
                data.columns = ['d__' + data.columns[i] for i in range(len(data.columns))]
                data_temp = data.reset_index().rename(columns={'index': 'sample_id'})
                meta_temp = meta.reset_index().rename(columns={'index': 'sample_id'})
                df_combined = pd.merge(data_temp, meta_temp, on='sample_id', how='inner').T
                df_combined.to_csv(f'./{output_folder}/dataset.csv', index=True, sep='\t', header=False)
                subset_data_cmd = f'python3 ./metaml_code/dataset_selection.py \
                                    ./metaml_results/dataset.csv \
                                    ./metaml_results/temp_dataset.txt \
                                    -z "d__" \
                                    -s Dataset:{dset}:{dset2} \
                                    -i Dataset:Group'
                classify_cmd = f'python3 ./metaml_code/classification.py \
                                            ./{output_folder}/temp_dataset.txt \
                                            ./{output_folder}/{split_type}_{disease}_{dset}_train_{dset2}_test \
                                            -z "d__" \
                                            -d 1:Group:1 \
                                            -g [] \
                                            -r 10 \
                                            -t Dataset:{dset2}'
                full_cmd = f"{subset_data_cmd} && {classify_cmd}"
                subprocess.run(full_cmd, shell=True, executable="/bin/bash")

    elif split_type == 'kfold':
        for dset in meta1['Dataset'].unique():
            if disease == 'crc':
                data, meta = utils.load_CRC_data(studies_to_include=[dset],
                                                 num_feat=150)
            elif disease == 'ibd':
                data, meta = utils.load_IBD_data(studies_to_include=[dset],
                                                 num_feat=150)
            elif disease == 't2d':
                data, meta = utils.load_T2D_data(studies_to_include=[dset],
                                                 num_feat=150)
            data.columns = ['d__' + data.columns[i] for i in range(len(data.columns))]
            data_temp = data.reset_index().rename(columns={'index': 'sample_id'})
            meta_temp = meta.reset_index().rename(columns={'index': 'sample_id'})
            df_combined = pd.merge(data_temp, meta_temp, on='sample_id', how='inner').T
            df_combined.to_csv(f'./{output_folder}/dataset.csv', index=True, sep='\t', header=False)
            subset_data_cmd = f'python3 ./metaml_code/dataset_selection.py \
                                            ./{output_folder}/dataset.csv \
                                            ./{output_folder}/temp_dataset.txt \
                                            -z "d__" \
                                            -s Dataset:{dset} \
                                            -i Group'
            classify_cmd = f'python3 ./metaml_code/classification.py \
                            ./{output_folder}/temp_dataset.txt \
                            ./{output_folder}/{split_type}_{disease}_{dset} \
                            -z "d__" \
                            -d 1:Group:1 \
                            -g [] \
                            -r 5 \
                            -p 5'
            full_cmd = f"{subset_data_cmd} && {classify_cmd}"
            subprocess.run(full_cmd, shell=True, executable="/bin/bash")

    # if disease == 'crc':
    #     data, meta = utils.load_CRC_data()
    # elif disease == 'ibd':
    #     data, meta = utils.load_IBD_data()
    # elif disease == 't2d':
    #     data, meta = utils.load_T2D_data()
    # data.columns = ['d__' + data.columns[i] for i in range(len(data.columns))]
    # data_temp = data.reset_index().rename(columns={'index': 'sample_id'})
    # meta_temp = meta.reset_index().rename(columns={'index': 'sample_id'})
    # df_combined = pd.merge(data_temp, meta_temp, on='sample_id', how='inner').T
    # df_combined.to_csv(f'./{output_folder}/dataset.csv', index=True, sep='\t', header=False)
    #
    # for d in meta['Dataset'].unique():
    #     if split_type == 'loso':
    #         subset_data_cmd = f'python3 ./metaml_code/dataset_selection.py \
    #                                         ./{output_folder}/dataset.csv \
    #                                         ./{output_folder}/temp_dataset.txt \
    #                                         -z "d__" \
    #                                         -i Dataset:Group'
    #         classify_cmd = f'python3 ./metaml_code/classification.py \
    #                         ./{output_folder}/temp_dataset.txt \
    #                         ./{output_folder}/{split_type}_{disease}_{d} \
    #                         -z "d__" \
    #                         -d 1:Group:1 \
    #                         -g [] \
    #                         -r 10 \
    #                         -t Dataset:{d}'
    #         full_cmd = f"{subset_data_cmd} && {classify_cmd}"
    #         subprocess.run(full_cmd, shell=True, executable="/bin/bash")
    #
    #     elif split_type == 'toso':
    #         for d2 in meta['Dataset'].unique():
    #             if d == d2:
    #                 continue
    #
    #             subset_data_cmd = f'python3 ./metaml_code/dataset_selection.py \
    #                                 ./metaml_results/dataset.csv \
    #                                 ./metaml_results/temp_dataset.txt \
    #                                 -z "d__" \
    #                                 -s Dataset:{d}:{d2} \
    #                                 -i Dataset:Group'
    #             classify_cmd = f'python3 ./metaml_code/classification.py \
    #                                         ./{output_folder}/temp_dataset.txt \
    #                                         ./{output_folder}/{split_type}_{disease}_{d}_train_{d2}_test \
    #                                         -z "d__" \
    #                                         -d 1:Group:1 \
    #                                         -g [] \
    #                                         -r 10 \
    #                                         -t Dataset:{d2}'
    #             full_cmd = f"{subset_data_cmd} && {classify_cmd}"
    #             subprocess.run(full_cmd, shell=True, executable="/bin/bash")
    #
    #     elif split_type == 'kfold':
    #         subset_data_cmd = f'python3 ./metaml_code/dataset_selection.py \
    #                                         ./{output_folder}/dataset.csv \
    #                                         ./{output_folder}/temp_dataset.txt \
    #                                         -z "d__" \
    #                                         -s Dataset:{d} \
    #                                         -i Group'
    #         classify_cmd = f'python3 ./metaml_code/classification.py \
    #                         ./{output_folder}/temp_dataset.txt \
    #                         ./{output_folder}/{split_type}_{disease}_{d} \
    #                         -z "d__" \
    #                         -d 1:Group:1 \
    #                         -g [] \
    #                         -r 5 \
    #                         -p 5'
    #         full_cmd = f"{subset_data_cmd} && {classify_cmd}"
    #         subprocess.run(full_cmd, shell=True, executable="/bin/bash")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get results for SIAMCAT and MetAML')
    parser.add_argument('--split_type', type=str, default='toso', help='kfold/loso/toso')
    parser.add_argument('--disease', type=str, default='crc', help='crc/ibd/t2d')

    args = parser.parse_args()
    split_type = args.split_type
    disease = args.disease

    run_siamcat(disease, split_type)
    run_metaml(disease, split_type)
