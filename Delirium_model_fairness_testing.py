import json
import os
import sys, argparse
import glob
import pickle

import numpy as np
import pandas as pd
import math
from pyarrow import feather  # directly writing import pyarrow didn't work
from sklearn.model_selection import train_test_split
from sklearn import svm, linear_model, model_selection, metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn import ensemble
import xgboost as xgb
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, RocCurveDisplay,PrecisionRecallDisplay,confusion_matrix
from datetime import datetime
import shap


class MyEncoder(json.JSONEncoder):
    # https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def normalization(data0, mode, normalizing_value, contin_var):
    data = data0.copy()
    if mode == 'mean_std':
        mean = normalizing_value['mean']
        std = normalizing_value['std']
        data[contin_var] = data[contin_var] - mean
        data[contin_var] = data[contin_var] / std
    if mode == 'min_max':
        min_v = normalizing_value['min']
        max_v = normalizing_value['max']
        data[contin_var] = data[contin_var] - min_v
        data[contin_var] = data[contin_var] / max_v
    return data

def preprocess_train_epic(preops, task, y_outcome=None):

    # imputing the bow data
    bow_cols = [col for col in preops.columns if 'bow' in col]

    preops['BOW_NA'] = np.where(np.isnan(preops[bow_cols[0]]), 1, 0)
    preops[bow_cols] = preops[bow_cols].fillna(0)


    lab_cats = pd.read_csv('/research2/ActFastData/Epic_TS_Prototyping/categories_labs.csv')
    # lab_cats = pd.read_csv('/input/categories_labs.csv')
    preop_labs_categorical = lab_cats['LAB_TEST'].unique()

    # dropping painful variables temporarily
    preops.drop(columns=['Cardiac Rhythm', 'CurrentMRN', 'AnestStart','orientation_numeric', 'SurgService_Value'], inplace=True)

    # ordinal_variables = ['LVEF', 'ASA', 'CHF_class', 'FunctionalCapacity', 'DiastolicFunction']
    ordinal_variables = ['LVEF', 'ASA', 'Pain Score', 'FunctionalCapacity', 'TOBACCO_USE', 'CHF_class', 'Gait/Transferring','DiastolicFunction', 'DyspneaFreq']

    categorical_variables = [i for i in preop_labs_categorical if i in preops.columns]
    binary_variables = []

    for a in preops.columns:
        if preops[a].dtype == 'bool':
            preops[a] = preops[a].astype('int32')
        if preops[a].dtype == 'int32' or preops[a].dtype == 'int64':
            if len(preops[a].unique()) < 10 and len(preops[a].unique()) > 2 and (a not in ordinal_variables):
                preops[a] = preops[a].astype('category')
                categorical_variables.append(a)
        if len(preops[a].unique()) <= 2 and (a not in ordinal_variables) and (a not in bow_cols):
            binary_variables.append(a)
        if preops[a].dtype == 'O' and (a not in binary_variables + ordinal_variables + bow_cols):
            # if (a not in binary_variables) and (a not in ordinal_variables) and (a not in bow_cols):
            preops[a] = preops[a].astype('category')
            categorical_variables.append(a)


    # following inf is more or less hardcoded based on how the data was at the training time.
    categorical_variables.append('SurgService_Name')
    # preops['SurgService_Value'].replace(['NULL', ''], [np.NaN, np.NaN], inplace=True) # name is sufficient
    preops['plannedDispo'].replace('', np.NaN, inplace=True)
    if False:
        preops['Cardiac Rhythm'].replace(list(preops['Cardiac Rhythm'].unique()),
                                     [v.split(";")[0] for v in list(preops['Cardiac Rhythm'].unique())], inplace=True)
        preops['Cardiac Rhythm'].replace('', np.NaN, inplace=True)
        # the following transition is being done because the above replacement changed the dtype of cardiac rhhythhm to 0 again and hence it was adding the variable again!!
        preops['Cardiac Rhythm'] = preops['Cardiac Rhythm'].astype('category')
    dif_dtype = [a for a in preops.columns if preops[a].dtype not in ['int32', 'int64', 'float64',
                                                                      'category'] and (a not in binary_variables + ordinal_variables + bow_cols)]  # columns with non-numeric datatype; this was used at the code development time
    for a in dif_dtype:
        preops[a] = preops[a].astype('category')
        categorical_variables.append(a)

    # this is kind of hardcoded; check your data beforehand for this
    preops['PlannedAnesthesia'].replace(
        [preops['PlannedAnesthesia'].unique()[0], preops['PlannedAnesthesia'].unique()[3]],
        np.NaN,
        inplace=True)  # this is done because there were two values for missing token (nan anf -inf)
    categorical_variables.append('PlannedAnesthesia')

    # remove if there are any duplicates in any of the variable name lists
    categorical_variables = [*set(categorical_variables)]

    continuous_variables = [i for i in preops.columns if
                            i not in (binary_variables + categorical_variables + ordinal_variables +bow_cols)]
    continuous_variables = [*set(continuous_variables)]

    categorical_variables.remove('orlogid')

    # rescaling the sex variable
    preops.loc[(preops.Sex == 1), 'Sex'] = 0
    preops.loc[(preops.Sex == 2), 'Sex'] = 1


    # one hot encoding
    preops_ohe = preops.copy()
    preops_ohe.drop(columns=categorical_variables, inplace=True)
    import itertools
    encoded_variables = list()
    for i in categorical_variables:
        temp = pd.get_dummies(preops[i], dummy_na=True, prefix=i)
        preops_ohe = pd.concat([preops_ohe, temp], axis=1)
        encoded_variables.append([column for column in temp.columns])
    encoded_variables = list(itertools.chain.from_iterable(encoded_variables))

    # partitioning the data into train, valid and test
    if task == 'postop_del':
        train0, test = train_test_split(preops_ohe, test_size=0.3, random_state=42, stratify=y_outcome['postop_del'])
        train, valid = train_test_split(train0, test_size=0.0005, random_state=42,
                                        stratify=y_outcome.iloc[train0.index]['postop_del'])

    train_index = train.index
    valid_index = valid.index
    test_index = test.index

    # print( list(train.columns) )
    train.drop(columns="orlogid", inplace=True)
    valid.drop(columns="orlogid", inplace=True)
    test.drop(columns="orlogid", inplace=True)

    meta_Data = {}
    # mean imputing and scaling the continuous variables

    train[continuous_variables].fillna(train[continuous_variables].mean(), inplace=True)  ## warning about copy
    valid[continuous_variables].fillna(train[continuous_variables].mean(), inplace=True)
    test[continuous_variables].fillna(train[continuous_variables].mean(), inplace=True)
    # this is done because nan that are of float type is not recognised as missing byt above commands
    for i in continuous_variables:
        if train[i].isna().any() == True or valid[i].isna().any() == True or test[i].isna().any() == True:
            train[i].replace(train[i].unique().min(), train[i].mean(), inplace=True)
            valid[i].replace(valid[i].unique().min(), train[i].mean(), inplace=True)
            test[i].replace(test[i].unique().min(), train[i].mean(), inplace=True)

    meta_Data["train_mean_cont"] = [train[i].mean() for i in continuous_variables]

    normalizing_values_cont = {}
    normalizing_values_cont['cont_names'] = continuous_variables
    normalizing_values_cont['mean'] = list(train[continuous_variables].mean(axis=0).values)
    normalizing_values_cont['std'] = list(train[continuous_variables].std(axis=0).values)
    normalizing_values_cont['min'] = list(train[continuous_variables].min(axis=0).values)
    normalizing_values_cont['max'] = list(train[continuous_variables].max(axis=0).values)
    train = normalization(train, 'mean_std', normalizing_values_cont, continuous_variables)
    valid = normalization(valid, 'mean_std', normalizing_values_cont, continuous_variables)
    test = normalization(test, 'mean_std', normalizing_values_cont, continuous_variables)
    meta_Data['norm_value_cont'] = normalizing_values_cont


    # median Imputing_ordinal variables

    # imputing
    for i in ordinal_variables:
        if np.isnan(preops[i].unique().min()) == True:
            train[i].replace(train[i].unique().min(), train[i].median(), inplace=True)
            valid[i].replace(valid[i].unique().min(), train[i].median(), inplace=True)
            test[i].replace(test[i].unique().min(), train[i].median(), inplace=True)

    meta_Data["train_median_ord"] = [train[i].median() for i in ordinal_variables]

    # normalizing
    normalizing_values_ord = {}
    normalizing_values_ord["ord_names"] = ordinal_variables
    normalizing_values_ord['mean'] = list(train[ordinal_variables].mean(axis=0).values)
    normalizing_values_ord['std'] = list(train[ordinal_variables].std(axis=0).values)
    normalizing_values_ord['min'] = list(train[ordinal_variables].min(axis=0).values)
    normalizing_values_ord['max'] = list(train[ordinal_variables].max(axis=0).values)

    train = normalization(train, 'mean_std', normalizing_values_ord, ordinal_variables)
    valid = normalization(valid, 'mean_std', normalizing_values_ord, ordinal_variables)
    test = normalization(test, 'mean_std', normalizing_values_ord, ordinal_variables)

    meta_Data['norm_value_ord'] = normalizing_values_ord

    train['postop_del'] = y_outcome.iloc[train_index]['postop_del']
    valid['postop_del'] = y_outcome.iloc[valid_index]['postop_del']
    test['postop_del'] = y_outcome.iloc[test_index]['postop_del']

    meta_Data["encoded_var"] = encoded_variables

    output_file_name = './Epic_preops_metadata.json'

    with open(output_file_name, 'w') as outfile:
        json.dump(meta_Data, outfile)

    return train, valid, test, train_index, valid_index, test_index

def preprocess_train_MV(preops, task, y_outcome=None):

    features_frm_freetxt = ['txwv' + str(i) for i in range(1, 51)]
    features_frm_diagncode = ['dxwv' + str(i) for i in range(1, 51)]

    some_cat_variables = preops.select_dtypes(include=['object']).columns
    dict_for_categorical_var = {}
    for i in some_cat_variables:
        temp = np.sort(preops[i].astype(str).unique())
        d = {v: j for j, v in enumerate(temp)}
        dict_for_categorical_var[i] = d

    print("Dictionary for object to categorical variables \n")
    print(dict_for_categorical_var)
    preops.replace(dict_for_categorical_var, inplace=True)

    # ordinal_variables = ['LVEF', 'ASA', 'FUNCTIONAL_CAPACITY', 'CHF_Diastolic_Function', 'StopBang_Total']
    ordinal_variables = [ 'ASA', 'StopBang_Total', 'CHF_Diastolic_Function']

    categorical_variables = [i for i in some_cat_variables if i in preops.columns and i not in ordinal_variables]
    binary_variables = []

    for a in preops.columns:
        if preops[a].dtype == 'bool':
            preops[a] = preops[a].astype('int32')
        if preops[a].dtype == 'int64':
            if len(preops[a].unique()) < 10 and len(preops[a].unique()) > 2 and (a not in ordinal_variables):
                preops[a] = preops[a].astype('category')
                categorical_variables.append(a)
        if len(preops[a].unique()) <= 2 and (a not in ordinal_variables):
            binary_variables.append(a)
        if preops[a].dtype == 'O':
            preops[a] = preops[a].astype('category')
            categorical_variables.append(a)


    categorical_variables = [*set(categorical_variables)]

    continuous_variables = [i for i in preops.columns if i not in features_frm_diagncode + features_frm_freetxt + categorical_variables + binary_variables + ordinal_variables]
    continuous_variables.remove('person_integer')

    # for var_name in preops.columns:
    #     if preops[var_name].dtype == 'int64' and var_name in continuous_variables:
    #         preops[var_name] = preops[var_name].astype('float')
    #     if preops[var_name].dtype == 'int64' and var_name in binary_variables+ordinal_variables:
    #         # preops[var_name] = preops[var_name].astype('int32')
    #         preops[var_name] = map(int, preops[var_name])



    preops_ohe = preops.copy()
    preops_ohe.drop(columns=categorical_variables, inplace=True)
    import itertools
    encoded_variables = list()
    for i in categorical_variables:
        temp = pd.get_dummies(preops[i], dummy_na=True, prefix=i)
        preops_ohe = pd.concat([preops_ohe, temp], axis=1)
        encoded_variables.append([column for column in temp.columns])
    encoded_variables = list(itertools.chain.from_iterable(encoded_variables))


    # partitioning the data into train, valid and test
    if task == 'postop_del':
        train0, test = train_test_split(preops_ohe, test_size=0.2, random_state=42, stratify=y_outcome['pdel'])
        train, valid = train_test_split(train0, test_size=0.15, random_state=42,
                                        stratify=y_outcome.iloc[train0.index]['pdel'])

    train_index = train.index
    valid_index = valid.index
    test_index = test.index

    # print( list(train.columns) )
    train.drop(columns="person_integer", inplace=True)
    valid.drop(columns="person_integer", inplace=True)
    test.drop(columns="person_integer", inplace=True)


    meta_Data = {}
    # mean imputing and scaling the continuous variables

    train[continuous_variables].fillna(train[continuous_variables].mean(), inplace=True)  ## warning about copy
    valid[continuous_variables].fillna(train[continuous_variables].mean(), inplace=True)
    test[continuous_variables].fillna(train[continuous_variables].mean(), inplace=True)
    # this is done because nan that are of float type is not recognised as missing byt above commands
    for i in continuous_variables:
        if train[i].isna().any() == True or valid[i].isna().any() == True or test[i].isna().any() == True:
            train[i].replace(train[i].unique().min(), train[i].mean(), inplace=True)
            valid[i].replace(valid[i].unique().min(), train[i].mean(), inplace=True)
            test[i].replace(test[i].unique().min(), train[i].mean(), inplace=True)

    meta_Data["train_mean_cont"] = [train[i].mean() for i in continuous_variables]

    normalizing_values_cont = {}
    normalizing_values_cont['cont_names'] = continuous_variables
    normalizing_values_cont['mean'] = list(train[continuous_variables].mean(axis=0).values)
    normalizing_values_cont['std'] = list(train[continuous_variables].std(axis=0).values)
    normalizing_values_cont['min'] = list(train[continuous_variables].min(axis=0).values)
    normalizing_values_cont['max'] = list(train[continuous_variables].max(axis=0).values)
    train = normalization(train, 'mean_std', normalizing_values_cont, continuous_variables)
    valid = normalization(valid, 'mean_std', normalizing_values_cont, continuous_variables)
    test = normalization(test, 'mean_std', normalizing_values_cont, continuous_variables)
    meta_Data['norm_value_cont'] = normalizing_values_cont


    # median Imputing_ordinal variables

    # imputing
    for i in ordinal_variables:
        if np.isnan(preops[i].unique().min()) == True:
            train[i].replace(train[i].unique().min(), train[i].median(), inplace=True)
            valid[i].replace(valid[i].unique().min(), train[i].median(), inplace=True)
            test[i].replace(test[i].unique().min(), train[i].median(), inplace=True)

    meta_Data["train_median_ord"] = [train[i].median() for i in ordinal_variables]

    # normalizing
    normalizing_values_ord = {}
    normalizing_values_ord["ord_names"] = ordinal_variables
    normalizing_values_ord['mean'] = list(train[ordinal_variables].mean(axis=0).values)
    normalizing_values_ord['std'] = list(train[ordinal_variables].std(axis=0).values)
    normalizing_values_ord['min'] = list(train[ordinal_variables].min(axis=0).values)
    normalizing_values_ord['max'] = list(train[ordinal_variables].max(axis=0).values)

    train = normalization(train, 'mean_std', normalizing_values_ord, ordinal_variables)
    valid = normalization(valid, 'mean_std', normalizing_values_ord, ordinal_variables)
    test = normalization(test, 'mean_std', normalizing_values_ord, ordinal_variables)


    meta_Data['norm_value_ord'] = normalizing_values_ord

    train['postop_del'] = y_outcome.iloc[train_index]['pdel']
    valid['postop_del'] = y_outcome.iloc[valid_index]['pdel']
    test['postop_del'] = y_outcome.iloc[test_index]['pdel']

    meta_Data["encoded_var"] = encoded_variables

    meta_Data["binary_var_name"] = binary_variables

    meta_Data["categorical_name"] = categorical_variables

    output_file_name = './MV_preops_metadata.json'

    with open(output_file_name, 'w') as outfile:
        json.dump(meta_Data, outfile, cls=MyEncoder)

    return train, valid, test, train_index, valid_index, test_index


task = 'postop_del'
data_gen = 1 # 1 only at the time of data processing and split else 0

if False:
    if data_gen == 1:

        # reading the files
        preops_file = pd.read_csv("/home/trips/data/preops.csv")
        outcome_file = pd.read_csv("/home/trips/data/outcomes.csv")

        preops_file = preops_file[(preops_file['neval_valid'] > 0) & (preops_file['blank_preop'] == 0)]
        preops_file.drop(columns=['Location', 'age_missing', 'year', 'case_duration', 'Intubation', 'blank_preop'],
                         inplace=True)

        unique_cases_preops = preops_file['caseid'].unique()
        unique_cases_outcome = outcome_file['caseid'].unique()
        common_unique_case_ids = list(set(unique_cases_outcome).intersection(unique_cases_preops))

        # dictionary creation etc to avoid alphanumeric caseids and replace them with a person integer
        preops = preops_file[preops_file['caseid'].isin(common_unique_case_ids)]
        outcome = outcome_file[outcome_file['caseid'].isin(common_unique_case_ids)]

        caseid_to_person_int_map = dict(zip(common_unique_case_ids, np.arange(len(common_unique_case_ids))))

        preops['person_integer'] = caseid_to_person_int_map.values()
        outcome['person_integer'] = caseid_to_person_int_map.values()

        preops.drop(columns=['caseid'], inplace=True)
        outcome.drop(columns=['caseid'], inplace=True)

        delirium_outcome = outcome[['person_integer', 'pdel']]
        na_cases = list(delirium_outcome[delirium_outcome['pdel'].isna() == True]['person_integer'])
        # Selecting only those cases where the outcome is measured
        delirium_outcome.drop(delirium_outcome[delirium_outcome['person_integer'].isin(na_cases)].index, inplace=True)
        preops.drop(preops[preops['person_integer'].isin(na_cases)].index, inplace=True)

        # resetting the index
        preops.reset_index(drop=True, inplace=True)
        delirium_outcome.reset_index(drop=True, inplace=True)

        del_predictors = ['person_integer', 'Anesthesia_Type', 'SPL_THEMES', 'RPL_THEMES', 'HTN', 'PHTN', 'CKD', 'COPD',
                          'OSA', 'DEMENTIA',
                          'StopBang_Total', 'StopBang_Observed', 'StopBang_Pressure', 'StopBang_Snore',
                          'StopBang_Tired',
                          'CPAP.Usage', 'Neck', 'Age', 'ASA', 'BMI', 'PAP_Type', 'Surg_Type', 'SEX', 'RACE',
                          'preop_los', 'CHF', 'CHF_Diastolic_Function', 'CAD', 'CAD_PRIORMI']

        preops_tr, preops_val, preops_te, tr_idx, val_idx, te_idx = preprocess_train_MV(preops[del_predictors], task,
                                                                                        delirium_outcome)

        # appending the person integer again to the data as it will be needed later when we use the TS data too
        preops_tr['person_integer'] = delirium_outcome.iloc[tr_idx]['person_integer']
        preops_val['person_integer'] = delirium_outcome.iloc[val_idx]['person_integer']
        preops_te['person_integer'] = delirium_outcome.iloc[te_idx]['person_integer']

        # saving the data split
        preops_tr.to_csv("./Validation_data_for_cards/MV_Original_train_" + str(task) + ".csv", index=False)
        preops_val.to_csv("./Validation_data_for_cards/MV_Original_valid_" + str(task) + ".csv", index=False)
        preops_te.to_csv("./Validation_data_for_cards/MV_Original_test_" + str(task) + ".csv", index=False)

    else:
        train = pd.read_csv("../Validation_data_for_cards/MV_Original_train_" + str(task) + ".csv")
        valid = pd.read_csv("../Validation_data_for_cards/MV_Original_valid_" + str(task) + ".csv")
        train.set_index('person_integer', inplace=True)
        valid.set_index('person_integer', inplace=True)

    # once you already have the data, the following is done to train the full and basic models on the saved train and validation data
    # model_list = ['LR', 'RF', 'DT', 'GBT', 'DNN', "XGB"]
    model_list = ["XGB"]


    """ FULL MODEL"""
    for model in model_list:
        if model == "LR":
            clf = linear_model.LogisticRegression(penalty='l2', n_jobs=-1)
        if model in ['RF']:
            clf = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1)
        if model in ["DT"]:
            clf = tree.DecisionTreeClassifier(max_depth=20)
        if model in ['GBT']:
            clf = ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=1)
        if model in ['DNN']:
            clf = MLPClassifier(random_state=1, hidden_layer_sizes=(256, 256), activation='relu', solver='sgd',
                                alpha=8e-5, batch_size=64, learning_rate_init=0.01, max_iter=50)
        if model in ['XGB']:
            clf = xgb.XGBClassifier()

        clf.fit(train.iloc[:, :-1].values, train.iloc[:, -1].to_frame().values)  # training with sensitive variable
        y_hat_valid = clf.predict_proba(valid.iloc[:, :-1].values)
        y_hat_valid = y_hat_valid[:, 1]

        # model performance on the validation set
        print("Model performance on the validation set for Delirium outcome with model: ", model)
        print("auroc: ", roc_auc_score(valid.iloc[:, -1].to_frame().values, y_hat_valid))
        print("auprc: ", average_precision_score(valid.iloc[:, -1].to_frame().values, y_hat_valid))

        valid.iloc[:, :-1].to_csv("./Validation_data_for_cards/MV_x_valid_" + str(model) + "_" + str(task) + ".csv")
        valid.iloc[:, -1].to_csv("./Validation_data_for_cards/MV_y_true_valid_" + str(model) + "_" + str(task) + ".csv")
        np.savetxt("./Validation_data_for_cards/MV_y_pred_prob_valid_Full_" + str(model) + "_" + str(task) + ".csv",
                   y_hat_valid, delimiter=",")

        # explanation for the fitted model on full feature set
        if model == "LR":
            explainer = shap.LinearExplainer(clf, train.iloc[:, :-1].values, feature_dependence="independent")
            shap_values = explainer.shap_values(valid.iloc[:, :-1].values)
        if model in ['GBT']:
            explainer = shap.TreeExplainer(model=clf, data=None, model_output='raw',
                                           feature_perturbation='tree_path_dependent')
            shap_values = explainer.shap_values(
                valid.iloc[:, :-1].values)  # the decision tree outputs two probabilities
        if model in ["DT", 'RF']:
            explainer = shap.TreeExplainer(model=clf, data=None, model_output='raw',
                                           feature_perturbation='tree_path_dependent')
            shap_values = explainer.shap_values(valid.iloc[:, :-1].values)[
                1]  # the decision tree outputs two probabilities
        if model in ['DNN']:
            back_data = shap.kmeans(train.iloc[:, :-1].values, 10).data
            explainer = shap.KernelExplainer(clf.predict_proba, back_data)
            shap_values = explainer.shap_values(valid.iloc[:, :-1].values)[1]

        np.savetxt("./Validation_data_for_cards/MV_Shap_on_x_valid_" + str(model) + "_" + str(task) + ".csv",
                   shap_values,
                   delimiter=",")

    """ SMALLER MODEL"""

    # to add race of it exists in the dataset
    basic_features = ['Surg_Type_0.0', 'Surg_Type_1.0', 'Surg_Type_2.0', 'Surg_Type_3.0', 'Surg_Type_4.0',
                      'Surg_Type_5.0',
                      'Surg_Type_6.0', 'Surg_Type_7.0', 'Surg_Type_8.0', 'Surg_Type_9.0', 'Surg_Type_10.0',
                      'Surg_Type_11.0',
                      'Surg_Type_12.0', 'Surg_Type_13.0', 'Surg_Type_14.0', 'Surg_Type_nan',
                      'SEX_0', 'SEX_1', 'SEX_2', 'SEX_nan', 'Age',
                      'RACE_0', 'RACE_1', 'RACE_2', 'RACE_3', 'RACE_4', 'RACE_nan']

    # subsetting the dataset
    x_train_few = train[basic_features].copy()
    x_test_few = valid[basic_features].copy()

    for model in model_list:
        if model == "LR":
            clf_level1 = linear_model.LogisticRegression(penalty='l2', n_jobs=-1)
        if model in ['RF']:
            clf_level1 = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1)
        if model in ["DT"]:
            clf_level1 = tree.DecisionTreeClassifier(max_depth=20)
        if model in ['GBT']:
            clf_level1 = ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=1)
        if model in ['DNN']:
            clf_level1 = MLPClassifier(random_state=1, hidden_layer_sizes=(256, 256), activation='relu', solver='sgd',
                                       alpha=8e-5, batch_size=64, learning_rate_init=0.01, max_iter=50)

        clf_level1.fit(x_train_few.values, train.iloc[:, -1].to_frame().values)
        y_hat_base = clf_level1.predict_proba(x_test_few.values)
        y_hat_base_valid = y_hat_base[:, 1]

        # model performance on the validation set
        print("Model performance on the validation set for Delirium outcome with model: ", model)
        print("auroc: ", roc_auc_score(valid.iloc[:, -1].to_frame().values, y_hat_base_valid))
        print("auprc: ", average_precision_score(valid.iloc[:, -1].to_frame().values, y_hat_base_valid))

        np.savetxt("./Validation_data_for_cards/MV_y_pred_prob_valid_Basic_" + str(model) + "_" + str(task) + ".csv",
                   y_hat_base_valid, delimiter=",")


if True:

    if data_gen==1:
        # reading the files
        # outcomes = feather.read_feather('/research2/ActFastData/Epic_TS_Prototyping/outcomes.feather')
        # outcomes_large = pd.read_csv('/research2/ActFastData/Actfast_deident_epic/epic_outcomes.csv')
        # preops = feather.read_feather('/research2/ActFastData/Epic_TS_Prototyping/preops.feather')

        # outcomes = feather.read_feather('/research2/ActFastData/Actfast_deident_epic/outcomes.feather')
        outcomes_large = pd.read_csv('/mnt/ris/ActFastData/Actfast_deident_epic/epic_outcomes.csv')
        preops = pd.read_csv('/mnt/ris/ActFastData/Actfast_deident_epic/epic_preop.csv')

        # prepping the delirium outcome dataframe
        delirium_outcome = pd.DataFrame(columns=['orlogid', 'postop_del'])
        delirium_outcome['orlogid'] = outcomes_large['orlogid']
        delirium_outcome['postop_del'] = outcomes_large['postop_del']
        delirium_outcome['postop_del'].value_counts(dropna=False)
        delirium_outcome['icu'] = outcomes_large['ICU']

        # preops.drop(preops[preops['RACE'].isin([2,3])].index, inplace=True) # this operation is being done here because after changing its type to categorical it becomes difficult to delete the category

        na_cases = list(delirium_outcome[delirium_outcome['postop_del'].isna() == True]['orlogid']) + list(preops[preops['RACE'].isin([2,3])]['orlogid'])
        # Selecting only those cases where the outcome is measured
        delirium_outcome.drop(delirium_outcome[delirium_outcome['orlogid'].isin(na_cases)].index, inplace=True)
        preops.drop(preops[preops['orlogid'].isin(na_cases)].index, inplace=True)


        # resetting the index
        preops.reset_index(drop=True, inplace=True)
        delirium_outcome.reset_index(drop=True, inplace=True)

        # preops processing
        preops_tr, preops_val, preops_te, tr_idx, val_idx, te_idx = preprocess_train_epic(preops, task, delirium_outcome)


        # appending the person integer again to the data as it will be needed later when we use the TS data too
        preops_tr['orlogid'] = delirium_outcome.iloc[tr_idx]['orlogid']
        preops_val['orlogid'] = delirium_outcome.iloc[val_idx]['orlogid']
        preops_te['orlogid'] = delirium_outcome.iloc[te_idx]['orlogid']

        # saving the data split
        preops_tr.to_csv("../Validation_data_for_cards/Epic_Original_train_" + str(task) + ".csv", index=False)
        preops_val.to_csv("../Validation_data_for_cards/Epic_Original_valid_" + str(task) + ".csv", index=False)
        preops_te.to_csv("../Validation_data_for_cards/Epic_Original_test_" + str(task) + ".csv", index=False)

    else:
        train = pd.read_csv("../Validation_data_for_cards/Epic_Original_train_" + str(task) + ".csv")
        # valid = pd.read_csv("../Validation_data_for_cards/Epic_Original_valid_" + str(task) + ".csv")
        valid = pd.read_csv("../Validation_data_for_cards/Epic_Original_test_" + str(task) + ".csv") # since the dataset is small so effectively partiitoning into two parts

        train.set_index('orlogid', inplace=True)
        valid.set_index('orlogid', inplace=True)

    # once you already have the data, the following is done to train the full and basic models on the saved train and validation data
    model_list = ['LR', 'RF', 'DT', 'GBT', 'DNN']

    """ FULL MODEL"""
    for model in model_list:
        if model == "LR":
            clf = linear_model.LogisticRegression(penalty='l2', n_jobs=-1)
        if model in ['RF']:
            clf = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1)
        if model in ["DT"]:
            clf = tree.DecisionTreeClassifier(max_depth=20)
        if model in ['GBT']:
            clf = ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=1)
        if model in ['DNN']:
            clf = MLPClassifier(random_state=1, hidden_layer_sizes=(256, 256), activation='relu', solver='sgd',
                                alpha=8e-5, batch_size=64, learning_rate_init=0.01, max_iter=50)

        clf.fit(train.iloc[:,:-1].values, train.iloc[:,-1].to_frame().values)  # training with sensitive variable
        y_hat_valid = clf.predict_proba(valid.iloc[:,:-1].values)
        y_hat_valid = y_hat_valid[:, 1]

        # model performance on the validation set
        print("Model performance on the validation set for Delirium outcome with model: ", model)
        print("auroc: ", roc_auc_score(valid.iloc[:,-1].to_frame().values, y_hat_valid))
        print("auprc: ", average_precision_score(valid.iloc[:,-1].to_frame().values, y_hat_valid))

        valid.iloc[:,:-1].to_csv("../Validation_data_for_cards/Epic_x_valid_" + str(model)+"_"+str(task)+ ".csv")
        valid.iloc[:,-1].to_csv("../Validation_data_for_cards/Epic_y_true_valid_" + str(model)+"_"+str(task)+ ".csv")
        np.savetxt("../Validation_data_for_cards/Epic_y_pred_prob_valid_Full_" + str(model)+"_"+str(task)+ ".csv",y_hat_valid, delimiter=",")

        # # explanation for the fitted model on full feature set
        # if model == "LR":
        #     explainer = shap.LinearExplainer(clf, train.iloc[:, :-1].values, feature_dependence="independent")
        #     shap_values = explainer.shap_values(valid.iloc[:, :-1].values)
        # if model in ['GBT']:
        #     explainer = shap.TreeExplainer(model=clf, data=None, model_output='raw',
        #                                    feature_perturbation='tree_path_dependent')
        #     shap_values = explainer.shap_values(valid.iloc[:, :-1].values)  # the decision tree outputs two probabilities
        # if model in ["DT", 'RF']:
        #     explainer = shap.TreeExplainer(model=clf, data=None, model_output='raw',
        #                                    feature_perturbation='tree_path_dependent')
        #     shap_values = explainer.shap_values(valid.iloc[:, :-1].values)[1]  # the decision tree outputs two probabilities
        # if model in ['DNN']:
        #     back_data = shap.kmeans(train.iloc[:, :-1].values, 10).data
        #     explainer = shap.KernelExplainer(clf.predict_proba, back_data)
        #     shap_values = explainer.shap_values(valid.iloc[:, :-1].values)[1]
        #
        # np.savetxt("./Validation_data_for_cards/Epic_Shap_on_x_valid_" + str(model) + "_" + str(task) + ".csv", shap_values,
        #            delimiter=",")

    """ SMALLER MODEL"""

    # to add race of it exists in the dataset
    basic_features_surg = [i for i in train.columns if 'SurgService_Name' in i]
    basic_features_race = [i for i in train.columns if 'RACE' in i]

    basic_features = basic_features_surg + basic_features_race + ['age', 'Sex']

    # subsetting the dataset
    x_train_few = train[basic_features].copy()
    x_test_few = valid[basic_features].copy()


    for model in model_list:
        if model == "LR":
            clf_level1 = linear_model.LogisticRegression(penalty='l2', n_jobs=-1)
        if model in ['RF']:
            clf_level1 = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1)
        if model in ["DT"]:
            clf_level1 = tree.DecisionTreeClassifier(max_depth=20)
        if model in ['GBT']:
            clf_level1 = ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=1)
        if model in ['DNN']:
            clf_level1 = MLPClassifier(random_state=1, hidden_layer_sizes=(256, 256), activation='relu', solver='sgd',
                                       alpha=8e-5, batch_size=64, learning_rate_init=0.01, max_iter=30)

        clf_level1.fit(x_train_few.values, train.iloc[:, -1].to_frame().values)
        y_hat_base = clf_level1.predict_proba(x_test_few.values)
        y_hat_base_valid = y_hat_base[:, 1]

        # model performance on the validation set
        print("Model performance on the validation set for Delirium outcome with model: ", model)
        print("auroc: ", roc_auc_score(valid.iloc[:, -1].to_frame().values, y_hat_base_valid))
        print("auprc: ", average_precision_score(valid.iloc[:, -1].to_frame().values, y_hat_base_valid))

        np.savetxt("../Validation_data_for_cards/Epic_y_pred_prob_valid_Basic_" + str(model) + "_" + str(task) + ".csv",
                   y_hat_base_valid, delimiter=",")


