import numpy as np
import pandas as pd

import utils


def sim_train_test(disease, train_study, test_study, alpha, proportion_diff, effect_size, num_samples=50):
    """
     returns simulated data set where training and validation data are a combination of train_study and test_study
     train_data = alpha * train_study + (1-alpha) * test_study
     test_data = test_data
     effect sizes between cases and controls are determined by effect_size,


    :param disease: disease to simulate
    :param train_study:
    :param test_study:
    :param alpha:
    :param proportion_diff: (0.05, 0.1, 0.15)
    :param effect_size:

    :return:
    """
    train_data, train_labels = [], []
    test_data, test_labels = [], []

    if disease == 'crc':
        train_study_data, train_study_meta = utils.load_CRC_data(studies_to_include=[train_study],
                                                                 transform=False,
                                                                 num_feat=150)
        test_study_data, test_study_meta = utils.load_CRC_data(studies_to_include=[train_study, test_study],
                                                               train_studies=[train_study],
                                                               transform=False,
                                                               num_feat=150)

    elif disease == 'ibd':
        train_study_data, train_study_meta = utils.load_IBD_data(studies_to_include=[train_study], transform=False,
                                                                 num_feat=150)
        test_study_data, test_study_meta = utils.load_IBD_data(studies_to_include=[train_study, test_study],
                                                               train_studies=[train_study],
                                                               transform=False,
                                                               num_feat=150)
    elif disease == 't2d':
        train_study_data, train_study_meta = utils.load_T2D_data(studies_to_include=[train_study], transform=False,
                                                                 num_feat=150)
        test_study_data, test_study_meta = utils.load_T2D_data(studies_to_include=[train_study, test_study],
                                                               train_studies=[train_study],
                                                               transform=False,
                                                               num_feat=150)

    train_study_meta = train_study_meta.loc[train_study_meta['Group'] == 0]
    train_study_data = train_study_data.loc[train_study_meta.index]
    train_control_vector = train_study_data.mean() + 1e-6
    train_control_vector /= train_control_vector.sum()

    test_study_meta = test_study_meta.loc[test_study_meta['Dataset'] == test_study]
    test_study_meta = test_study_meta.loc[test_study_meta['Group'] == 0]
    test_study_data = test_study_data.loc[test_study_meta.index]
    test_control_vector = test_study_data.mean() + 1e-6
    test_control_vector /= test_control_vector.sum()

    train_control_vector = alpha * train_control_vector + (1 - alpha) * test_control_vector

    # Generate control samples
    rng = np.random.default_rng()
    train_controls = np.array([
        rng.multinomial(1e6, np.random.dirichlet(train_control_vector * 10))
        for _ in range(num_samples)
    ])
    test_controls = np.array([
        rng.multinomial(1e6, np.random.dirichlet(test_control_vector * 10))
        for _ in range(num_samples)
    ])

    # get differential OTU indices
    diff_idx = np.random.permutation(np.arange(1, len(train_control_vector)))
    num_to_change = int((proportion_diff * len(train_control_vector)) / 2)
    depleted_idx = diff_idx[:num_to_change]
    enriched_idx = diff_idx[num_to_change: num_to_change * 2]

    # generate case samples
    train_case_vector = train_control_vector.copy()
    train_case_vector[depleted_idx] /= effect_size
    train_case_vector[enriched_idx] *= effect_size
    train_case_vector /= train_case_vector.sum()

    test_case_vector = test_control_vector.copy()
    test_case_vector[depleted_idx] /= effect_size
    test_case_vector[enriched_idx] *= effect_size
    test_case_vector /= test_case_vector.sum()

    train_cases = np.array([
        rng.multinomial(1e6, np.random.dirichlet(train_case_vector * 10))
        for _ in range(num_samples)
    ])
    test_cases = np.array([
        rng.multinomial(1e6, np.random.dirichlet(test_case_vector * 10))
        for _ in range(num_samples)
    ])

    # return as pd dataframes
    data = pd.DataFrame(columns=[f'OTU_{i}' for i in range(len(train_control_vector))],
                        index=[f'sample_{i}' for i in range(4 * num_samples)])
    data[:] = np.vstack([train_controls, train_cases, test_controls, test_cases])
    meta = pd.DataFrame(columns=['Dataset', 'Group'],
                        index=[f'sample_{i}' for i in range(4 * num_samples)])
    meta['Dataset'] = ['train'] * (2 * num_samples) + ['test'] * (2 * num_samples)
    meta['Group'] = [0] * num_samples + [1] * num_samples + [0] * num_samples + [1] * num_samples

    data = data.fillna(0.0)
    return data, meta


def convert_data(data, transform='relative_abundance'):
    if transform == 'relative_abundance':
        data = data.div(data.sum(axis=1), axis=0)
    elif transform == 'clr':
        data = data.div(data.sum(axis=1), axis=0)
        data = data + 1e-5
        data = utils.clr_transform(data)
    return data.fillna(0.0)


# d, m = sim_train_test('crc', 'FengQ_2015', 'GuptaA_2019', alpha,
#                    proportion_diff, effect_size)
