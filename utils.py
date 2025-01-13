import numpy as np
import pandas as pd


def load_CRC_data(studies=None, relative_abundance=True, clr=True, num_feat=250):
    """

    :param studies:
    :param relative_abundance:
    :param clr:
    :param num_feat: number of each study's most abundant features to keep
    :return:
    """
    if studies is None:
        studies = ['feng', 'hannigan', 'thomas', 'vogtmann', 'yu', 'zeller']
    meta = pd.DataFrame()
    data = pd.DataFrame()
    sample_num = 0
    for study in studies:
        tmp = pd.read_csv('./CRC_data/{}_data.csv'.format(study), index_col=0, low_memory=False)
        sample_names = ['sample_{}'.format(sample_num + i) for i in range(len(tmp))]
        sample_num += len(tmp)
        tmp.index = sample_names
        data = pd.concat([data, tmp.iloc[:, :-1]]).fillna(0.0)

        tmp_meta = pd.DataFrame(index=sample_names)
        tmp_meta['Dataset'] = [study]*len(tmp_meta)
        tmp_meta['Condition'] = tmp['status']
        tmp_meta['Group'] = [0 if tmp['status'][i] == 'control' else 1 for i in range(len(tmp['status']))]
        meta = pd.concat([meta, tmp_meta])
    data = data.astype(float)
    data = data.loc[meta.index, :]
    selected_cols = set()
    for dataset in meta['Dataset'].unique():
        dataset_indices = meta[meta['Dataset'] == dataset].index
        top_cols = data.loc[dataset_indices].var().nlargest(num_feat).index
        selected_cols.update(top_cols)
    cols = data.columns.isin(selected_cols)
    data = data.loc[:, cols]
    if relative_abundance:
        data = data.div(data.sum(axis=1), axis=0)
        meta = meta.loc[data.index, :]
    if clr:
        data = data + 1e-5
        geometric_means = np.exp(np.log(data).mean(axis=1))
        data = np.log(data.divide(geometric_means, axis=0))
    data = data.fillna(0.0)
    return data, meta


def metric_from_confusion_matrix(conf_matrix, metric='mcc'):
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

