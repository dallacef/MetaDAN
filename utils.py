import numpy as np
import pandas as pd


def load_CRC_data(studies=None, relative_abundance=True, clr=True):
    if studies is None:
        studies = ['feng', 'hannigan', 'thomas', 'vogtmann', 'yu', 'zeller']
    meta = pd.DataFrame()
    data = pd.DataFrame()
    sample_num = 0
    for study in studies:
        tmp = pd.read_csv('./FMT_Data/CRC_data/{}_data.csv'.format(study), index_col=0, low_memory=False)
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
        top_cols = data.loc[dataset_indices].var().nlargest(250).index
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
