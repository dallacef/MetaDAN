import numpy as np
import utils

def sim_train_test_val(disease, train_study, test_study, num_features, alpha, num_diff, effect_size):
    """
    returns simulated data set where training and validation data are a combination of train_study and test_study
     train_data = alpha * train_study + (1-alpha) * test_study
     val_data = alpha * train_study + (1-alpha) * test_study
     test_data = test_data
     effect sizes between cases and controls are determined by effect_size,


    :param disease: disease to simulate
    :param train_study:
    :param test_study:
    :param num_features:
    :param alpha:
    :param num_diff:
    :param effect_size:

    :return:
    """
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    val_data, val_labels = [], []

    # todo load training study and reduce to controls only
    if disease == 'crc':
        training_study, training_meta = utils.load_CRC_data(studies_to_include=[train_study], transform=False, num_feat=num_features)
    elif disease == 'ibd':
        training_study, training_meta = utils.load_IBD_data(studies_to_include=[train_study], transform=False, num_feat=num_features)
    elif disease == 't2d':
        training_study, training_meta = utils.load_T2D_data(studies_to_include=[train_study], transform=False, num_feat=num_features)

    # todo get mean vector and normalize to get probabilities

    # todo load testing study and reduce to controls only

    # todo reduce to match training set features

    # todo adjust training mean vector based on alpha

    # todo generate training controls

    # todo generate validation controls

    # todo get pos_diff_OTUs and neg_diff_OTUs based on num_diff

    # todo generate training cases based on pos_diff_OTUs and neg_diff_OTUs and effect_size

    # todo generate validation cases based on pos_diff_OTUs and neg_diff_OTUs and effect_size

    # todo generate testing controls

    # todo generate testing cases based on pos_diff_OTUs and neg_diff_OTUs and effect_size

    # todo return all along with labels

    # todo fix all other methods runs to ensure data loading is correct (hint: its not)

    ...