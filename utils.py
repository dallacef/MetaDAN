import numpy as np
import pandas as pd
from scipy.stats import gmean


def load_IBD_data(studies_to_include=None, transform=None, num_feat=-1, train_studies=None, rel_abun=True):
    """
    :param studies_to_include:
    :param transform:
    :param num_feat: number of each study's most abundant features to keep
    :param train_studies:
    :param rel_abun:
    :return:
    """
    data = pd.read_csv('curated_data/ibd_data.csv', header=0, index_col=0)
    new_sample_names = {data.index[i]: f'sample_{i}' for i in range(data.shape[0])}
    data.rename(index=new_sample_names, inplace=True)

    meta = pd.read_csv('curated_data/ibd_meta.csv', header=0, index_col=2).iloc[:, 1:]
    meta.rename(index=new_sample_names, inplace=True)
    if studies_to_include is None:
        studies_to_include = meta['study_name'].unique()
    meta = meta.loc[meta['study_name'].isin(studies_to_include)]

    idx_to_keep = []
    idx_to_remove = []
    # HMP_2019_ibdmdb
    if 'HMP_2019_ibdmdb' in studies_to_include:
        idx_to_keep.extend(
            list(
                meta.loc[
                    (meta['study_name'] == 'HMP_2019_ibdmdb') &
                    (meta['visit_number'] == 1)
                    ].index
            )
        )
        idx_to_remove.extend(['sample_299', 'sample_657', 'sample_767', 'sample_1419', 'sample_1420', 'sample_1464'])

        # IjazUZ_2017
    if 'IjazUZ_2017' in studies_to_include:
        idx_to_keep.extend(
            list(
                meta.loc[
                    (meta['study_name'] == 'IjazUZ_2017') &
                    (meta['study_condition'] == 'control')
                    ].index
            )
        )
        idx_to_keep.extend(
            list(
                meta.loc[
                    (meta['study_name'] == 'IjazUZ_2017') &
                    (meta['days_from_first_collection'] == 0)
                    ].index
            )
        )
    # NielsenHB_2014
    if 'NielsenHB_2014' in studies_to_include:
        idx_to_keep.extend(
            list(
                meta.loc[
                    (meta['study_name'] == 'NielsenHB_2014') &
                    (meta['days_from_first_collection'] == 0)
                    ].index
            )
        )
    meta = meta.loc[idx_to_keep]
    meta = meta.drop(idx_to_remove)
    meta.rename(columns={
        "study_name": "Dataset",
        "study_condition": 'Condition'
    }, inplace=True)
    meta = meta.loc[:, ('Dataset', 'Condition')]
    meta['Group'] = [0 if meta['Condition'][i] == 'control' else 1 for i in range(len(meta['Condition']))]
    # realign data index to meta's
    data = data.loc[meta.index, :]
    if rel_abun:
        data = data.div(data.sum(axis=1), axis=0)
    if num_feat != -1:
        selected_cols = set()
        if train_studies is None:
            train_studies = meta['Dataset'].unique()
        for dataset in train_studies:
            dataset_indices = meta[meta['Dataset'] == dataset].index
            top_cols = data.loc[dataset_indices].var().nlargest(num_feat).index
            selected_cols.update(top_cols)
        cols = data.columns.isin(selected_cols)
        data = data.loc[:, cols]
    if transform == 'clr':
        data = data + 1e-5
        data = clr_transform(data)
    elif transform == 'ilr':
        data = data + 1e-5
        data = ilr_transform(data)
    data = data.fillna(0.0)
    return data, meta


def load_CRC_data(studies_to_include=None, transform=None, num_feat=-1, train_studies=None, keep_adenoma=False,
                  rel_abun=True):
    """
    :param studies_to_include:
    :param transform:
    :param num_feat: number of each study's most abundant features to keep
    :param train_studies:
    :param keep_adenoma:
    :param rel_abun:
    :return:
    """
    data = pd.read_csv('curated_data/crc_data.csv', header=0, index_col=0)
    new_sample_names = {data.index[i]: f'sample_{i}' for i in range(data.shape[0])}
    data.rename(index=new_sample_names, inplace=True)

    meta = pd.read_csv('curated_data/crc_meta.csv', header=0, index_col=2).iloc[:, 1:]
    meta.rename(index=new_sample_names, inplace=True)
    if studies_to_include is None:
        studies_to_include = meta['study_name'].unique()
    meta = meta.loc[meta['study_name'].isin(studies_to_include)]

    idx_to_keep = []
    idx_to_remove = []
    # FengQ_2015
    if 'FengQ_2015' in studies_to_include:
        idx_to_keep.extend(
            list(
                meta.loc[meta['study_name'] == 'FengQ_2015'].index
            )
        )
    # GuptaA_2019
    if 'GuptaA_2019' in studies_to_include:
        idx_to_keep.extend(
            list(
                meta.loc[meta['study_name'] == 'GuptaA_2019'].index
            )
        )
    # HanniganGD_2017
    if 'HanniganGD_2017' in studies_to_include:
        idx_to_keep.extend(
            list(
                meta.loc[meta['study_name'] == 'HanniganGD_2017'].index
            )
        )
    # ThomasAM_2019_c
    if 'ThomasAM_2019_c' in studies_to_include:
        idx_to_keep.extend(
            list(
                meta.loc[meta['study_name'] == 'ThomasAM_2019_c'].index
            )
        )
    # VogtmannE_2016
    if 'VogtmannE_2016' in studies_to_include:
        idx_to_keep.extend(
            list(
                meta.loc[meta['study_name'] == 'VogtmannE_2016'].index
            )
        )
        meta.loc[(meta['study_name'] == 'VogtmannE_2016') &
                 (meta['disease'] == 'healthy'), 'study_condition'] = 'control'
    # WirbelJ_2018
    if 'WirbelJ_2018' in studies_to_include:
        idx_to_keep.extend(
            list(
                meta.loc[meta['study_name'] == 'WirbelJ_2018'].index
            )
        )
    # YachidaS_2019
    if 'YachidaS_2019' in studies_to_include:
        idx_to_keep.extend(
            list(
                meta.loc[(meta['study_name'] == 'YachidaS_2019') &
                         (meta['study_condition'] != 'carcinoma_surgery_history')].index
            )
        )
    # YuJ_2015
    if 'YuJ_2015' in studies_to_include:
        idx_to_keep.extend(
            list(
                meta.loc[meta['study_name'] == 'YuJ_2015'].index
            )
        )
    # ZellerG_2014
    if 'ZellerG_2014' in studies_to_include:
        idx_to_keep.extend(
            list(
                meta.loc[meta['study_name'] == 'ZellerG_2014'].index
            )
        )

    meta = meta.loc[idx_to_keep]
    meta = meta.drop(idx_to_remove)
    if not keep_adenoma:
        meta = meta.loc[meta['study_condition'] != 'adenoma']

    meta.rename(columns={
        "study_name": "Dataset",
        "study_condition": 'Condition'
    }, inplace=True)
    meta = meta.loc[:, ('Dataset', 'Condition')]
    meta['Group'] = [0 if meta['Condition'][i] == 'control' else 1 for i in range(len(meta['Condition']))]
    # realign data index to meta's
    data = data.loc[meta.index, :]
    if rel_abun:
        data = data.div(data.sum(axis=1), axis=0)
    if num_feat != -1:
        selected_cols = set()
        if train_studies is None:
            train_studies = meta['Dataset'].unique()
        for dataset in train_studies:
            dataset_indices = meta[meta['Dataset'] == dataset].index
            top_cols = data.loc[dataset_indices].var().nlargest(num_feat).index
            selected_cols.update(top_cols)
        cols = data.columns.isin(selected_cols)
        data = data.loc[:, cols]
    if transform == 'clr':
        data = data + 1e-5
        data = clr_transform(data)
    elif transform == 'ilr':
        data = data + 1e-5
        data = ilr_transform(data)
    data = data.fillna(0.0)
    return data, meta


def load_T2D_data(studies_to_include=None, transform=None, num_feat=-1, train_studies=None, keep_igt=False,
                  rel_abun=True):
    """
    :param studies_to_include:
    :param transform:
    :param num_feat: number of each study's most abundant features to keep
    :param train_studies:
    :param keep_igt:
    :param rel_abun:
    :return:
    """
    data = pd.read_csv('curated_data/t2d_data.csv', header=0, index_col=0)
    new_sample_names = {data.index[i]: f'sample_{i}' for i in range(data.shape[0])}
    data.rename(index=new_sample_names, inplace=True)

    meta = pd.read_csv('curated_data/t2d_meta.csv', header=0, index_col=2).iloc[:, 1:]
    meta.rename(index=new_sample_names, inplace=True)
    if studies_to_include is None:
        studies_to_include = meta['study_name'].unique()
    meta = meta.loc[meta['study_name'].isin(studies_to_include)]

    idx_to_keep = []
    idx_to_remove = []
    # KarlssonFH_2013
    if 'KarlssonFH_2013' in studies_to_include:
        idx_to_keep.extend(
            list(
                meta.loc[meta['study_name'] == 'KarlssonFH_2013'].index
            )
        )
    # QinJ_2012
    if 'QinJ_2012' in studies_to_include:
        idx_to_keep.extend(
            list(
                meta.loc[meta['study_name'] == 'QinJ_2012'].index
            )
        )
        meta.loc[(meta['study_name'] == 'QinJ_2012') &
                 (meta['disease'] == 'healthy'), 'study_condition'] = 'control'

    meta = meta.loc[idx_to_keep]
    meta = meta.drop(idx_to_remove)
    if not keep_igt:
        meta = meta.loc[meta['study_condition'] != 'IGT']

    meta.rename(columns={
        "study_name": "Dataset",
        "study_condition": 'Condition'
    }, inplace=True)
    meta = meta.loc[:, ('Dataset', 'Condition')]
    meta['Group'] = [0 if meta['Condition'][i] == 'control' else 1 for i in range(len(meta['Condition']))]
    # realign data index to meta's
    data = data.loc[meta.index, :]
    if rel_abun:
        data = data.div(data.sum(axis=1), axis=0)
    if num_feat != -1:
        selected_cols = set()
        if train_studies is None:
            train_studies = meta['Dataset'].unique()
        for dataset in train_studies:
            dataset_indices = meta[meta['Dataset'] == dataset].index
            top_cols = data.loc[dataset_indices].var().nlargest(num_feat).index
            selected_cols.update(top_cols)
        cols = data.columns.isin(selected_cols)
        data = data.loc[:, cols]
    if transform == 'clr':
        data = data + 1e-5
        data = clr_transform(data)
    elif transform == 'ilr':
        data = data + 1e-5
        data = ilr_transform(data)
    data = data.fillna(0.0)
    return data, meta


def clr_transform(df):
    """
    Perform Centered Log-Ratio (CLR) transformation on a Pandas DataFrame.

    Parameters:
    - df: Pandas DataFrame (n_samples, n_features) where rows are samples and columns are features.

    Returns:
    - Pandas DataFrame of CLR-transformed values with the same shape.
    """
    if (df <= 0).any().any():
        raise ValueError("All elements in the DataFrame must be positive for log transformations.")

    geometric_mean = gmean(df, axis=1)
    clr_values = np.log(df.div(geometric_mean, axis=0))

    return pd.DataFrame(clr_values, index=df.index, columns=df.columns)


def ilr_transform(df):
    """
    Perform Isometric Log-Ratio (ILR) transformation on a Pandas DataFrame.

    Parameters:
    - df: Pandas DataFrame (n_samples, n_features) where rows are samples and columns are features.

    Returns:
    - Pandas DataFrame of ILR-transformed values with shape (n_samples, n_features - 1).
    """
    if (df <= 0).any().any():
        raise ValueError("All elements in the DataFrame must be positive for log transformations.")

    n_components = df.shape[1]
    ilr_values = np.zeros((df.shape[0], n_components - 1))

    for j in range(n_components - 1):
        g_j = gmean(df.iloc[:, :j + 1], axis=1)  # Geometric mean of first j+1 features
        ilr_values[:, j] = np.sqrt((j + 1) / (j + 2)) * np.log(g_j / df.iloc[:, j + 1])

    ilr_columns = [f"ILR_{i + 1}" for i in range(n_components - 1)]
    return pd.DataFrame(ilr_values, index=df.index, columns=ilr_columns)


def metric_from_confusion_matrix(conf_matrix, metric='mcc'):
    if conf_matrix is None:
        return None
    # Extract values from confusion matrix
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    if metric == "acc":
        # Calculate Accuracy
        if (TP + TN + FP + FN) == 0:
            return 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        return accuracy

    elif metric == "f1":
        # Calculate F1 Score
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return f1

    elif metric == "mcc":
        # Calculate MCC
        numerator = (TP * TN) - (FP * FN)
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = numerator / denominator if denominator != 0 else 0
        mcc = (mcc + 1) / 2
        return mcc

    elif metric == "tpr":
        # Calculate TPR
        if (TP + FN) == 0:
            return 0
        tpr = TP / (TP + FN)
        return tpr

    elif metric == "fpr":
        # Calculate FPR
        if (FP + TN) == 0:
            return 0
        tpr = FP / (FP + TN)
        return tpr
