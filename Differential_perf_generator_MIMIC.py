import numpy as np
from sklearn import svm, linear_model, model_selection, metrics
from sklearn import tree
from sklearn import ensemble
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import pickle
import os.path
import math
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import random
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
# from fpdf import FPDF
import re
import shap
import scipy
import datetime
import json


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


def Match(groups, propensity, caliper=0.1, caliper_method="propensity", replace=False):
    '''
    Implements greedy one-to-one matching on propensity scores.

    Inputs:
    groups = Array-like object of treatment assignments.  Must be 2 groups
    propensity = Array-like object containing propensity scores for each observation. Propensity and groups should be in the same order (matching indices)
    caliper = a numeric value, specifies maximum distance (difference in propensity scores or SD of logit propensity)
    caliper_method = a string: "propensity" (default) if caliper is a maximum difference in propensity scores,
            "logit" if caliper is a maximum SD of logit propensity, or "none" for no caliper
    replace = Logical for whether individuals from the larger group should be allowed to match multiple individuals in the smaller group.
        (default is False)

    Output:
    A series containing the individuals in the control group matched to the treatment group.
    Note that with caliper matching, not every treated individual may have a match.
    '''

    # Check inputs
    if any(propensity <= 0) or any(propensity >= 1):
        raise ValueError('Propensity scores must be between 0 and 1')
    elif not (0 <= caliper < 1):
        if caliper_method == "propensity" and caliper > 1:
            raise ValueError('Caliper for "propensity" method must be between 0 and 1')
        elif caliper < 0:
            raise ValueError('Caliper cannot be negative')
    elif len(groups) != len(propensity):
        raise ValueError('groups and propensity scores must be same dimension')
    elif len(groups.unique()) != 2:
        raise ValueError('wrong number of groups: expected 2')

    # # Transform the propensity scores and caliper when caliper_method is "logit" or "none"
    # if caliper_method == "logit":
    #     propensity = log(propensity / (1 - propensity))
    #     caliper = caliper * np.std(propensity)
    # elif caliper_method == "none":
    #     caliper = 0

    # Code groups as 0 and 1
    groups = groups == groups.unique()[0]
    N = len(groups)
    N1 = groups[groups == 1].index;
    N2 = groups[groups == 0].index
    g1, g2 = propensity[groups == 1], propensity[groups == 0]
    # Check if treatment groups got flipped - the smaller should correspond to N1/g1
    if len(N1) > len(N2):
        N1, N2, g1, g2 = N2, N1, g2, g1

    # Randomly permute the smaller group to get order for matching
    morder = np.random.permutation(N1)
    matches = {}

    for m in morder:
        dist = abs(g1[m] - g2)
        if (dist.min() <= caliper) or not caliper:
            matches[m] = dist.idxmin()  # Potential problem: check for ties
            if not replace:
                g2 = g2.drop(matches[m])

    # for m in morder:
    #     dist = abs(g1[m] - g2)
    #     if dist.min() <= caliper:
    #         matches[m] = dist.idxmin()
    #         g2 = g2.drop(matches[m])
    return (matches)


def Prop_score_comp(features_data, index_levels, features, ohe_sense_data):
    sens_var_ohe = list(index_levels.keys())
    cls = linear_model.LogisticRegression(random_state=42)
    # calibration of the classifier
    cls = CalibratedClassifierCV(cls)
    if sens_var_ohe[0].split("_")[0] == 'Sex':
        X_pre = features_data[features]
        # X_pre.drop(columns=['HEIGHT_IN_INCHES'], inplace=True)  # do not use height as one of the independent variables
        y_pre = np.where(ohe_sense_data[sens_var_ohe[0].split("_")[0] + "_1"] == 1, 1, 0)
    if sens_var_ohe[0].split("_")[0] == 'RACE':
        # first pair
        x_pre0 = features_data[features][
            features_data[features].index.isin(index_levels[sens_var_ohe[1]] + index_levels[sens_var_ohe[2]])]
        # if data_version == 'MIMICwithsens':
        #     x_pre0.drop(columns=['RACE_1'], inplace=True)
        y_pre0 = ohe_sense_data.loc[x_pre0.index]['RACE_1']  # 1 and 0


        cls.fit(x_pre0, y_pre0)

        propensity_BlacktoWhite = cls.predict_proba(x_pre0)[:, 1]
        x_pre0['prop_BlacktoWhite'] = propensity_BlacktoWhite

        # matching method
        after_matching_BlacktoWhite = Match(y_pre0, x_pre0.prop_BlacktoWhite)

        te_indixes_BlacktoWhite = list(after_matching_BlacktoWhite.keys()) + list(after_matching_BlacktoWhite.values())

        # -------------------------------------------------------------------------------------
        # second pair
        x_pre0 = features_data[features][
            features_data[features].index.isin(index_levels[sens_var_ohe[0]] + index_levels[sens_var_ohe[2]])]
        y_pre0 = ohe_sense_data.loc[x_pre0.index]['RACE_-1']  # 1 and 0

        cls.fit(x_pre0, y_pre0)

        propensity_BlacktoOthers = cls.predict_proba(x_pre0)[:, 1]
        x_pre0['prop_BlacktoOthers'] = propensity_BlacktoOthers
        # matching method
        after_matching_BlacktoOthers = Match(y_pre0, x_pre0.prop_BlacktoOthers)

        te_indixes_BlacktoOthers = list(after_matching_BlacktoOthers.keys()) + list(
            after_matching_BlacktoOthers.values())

        # -----------------------------------------------------------------------------------
        # third pair
        x_pre0 = features_data[features][
            features_data[features].index.isin(index_levels[sens_var_ohe[0]] + index_levels[sens_var_ohe[1]])]
        y_pre0 = ohe_sense_data.loc[x_pre0.index]['RACE_-1']  # 1 and 0

        cls.fit(x_pre0, y_pre0)

        propensity_WhitetoOthers = cls.predict_proba(x_pre0)[:, 1]
        x_pre0['prop_WhitetoOthers'] = propensity_WhitetoOthers
        # matching method
        after_matching_WhitetoOthers = Match(y_pre0, x_pre0.prop_WhitetoOthers)

        te_indixes_WhitetoOthers = list(after_matching_WhitetoOthers.keys()) + list(
            after_matching_WhitetoOthers.values())

        return te_indixes_BlacktoWhite, te_indixes_BlacktoOthers, te_indixes_WhitetoOthers
    if sens_var_ohe[0].split("_")[0] == "age":
        # first pair
        x_pre0 = features_data[features][
            features_data[features].index.isin(index_levels[sens_var_ohe[0]] + index_levels[sens_var_ohe[1]])]
        y_pre0 = ohe_sense_data.loc[x_pre0.index]['age_1.0']  # 1 and 0

        cls.fit(x_pre0, y_pre0)

        propensity_1to0 = cls.predict_proba(x_pre0)[:, 1]
        x_pre0['prop_1to0'] = propensity_1to0
        # matching method
        after_matching_1to0 = Match(y_pre0, x_pre0.prop_1to0)

        te_indixes_1to0 = list(after_matching_1to0.keys()) + list(after_matching_1to0.values())

        # Second pair
        x_pre2 = features_data[features][
            features_data[features].index.isin(index_levels[sens_var_ohe[2]] + index_levels[sens_var_ohe[1]])]
        y_pre2 = ohe_sense_data.loc[x_pre2.index]['age_1.0']  # 1 and 2

        cls.fit(x_pre2, y_pre2)

        propensity_1to2 = cls.predict_proba(x_pre2)[:, 1]
        x_pre2['prop_1to2'] = propensity_1to2
        # matching method
        after_matching_1to2 = Match(y_pre2, x_pre2.prop_1to2)

        te_indixes_1to2 = list(after_matching_1to2.keys()) + list(after_matching_1to2.values())

        return te_indixes_1to0, te_indixes_1to2

    else:
        cls.fit(X_pre, y_pre)

        propensity = cls.predict_proba(X_pre)[:, 1]
        features_data['prop'] = propensity
        features_data['sens_Var_to_match'] = y_pre
        # matching method
        after_matching = Match(features_data['sens_Var_to_match'], features_data.prop)

        te_indixes = list(after_matching.keys()) + list(after_matching.values())
        return te_indixes


def Model_Report_card(validation_pred, validation_x, validation_true_y,
                      sens_var, task, MIMIC_sens_ohe):  # valid_pred need not be discrete, sens_var is a list
    features = list(validation_x.columns)
    features.remove('hadm_id')
    df_of_all_orig = validation_x.copy()
    df_of_all_orig[task] = validation_true_y[task].copy()
    df_of_all_orig['pred_prob_y'] = validation_pred.copy()
    df_of_all_orig.set_index('hadm_id', inplace=True)
    MIMIC_sens_ohe.set_index('hadm_id', inplace=True)

    true_outcome_rate = len(df_of_all_orig[df_of_all_orig[task] == 1]) / len(df_of_all_orig)
    threshold = Find_Optimal_Cutoff(df_of_all_orig[task], df_of_all_orig['pred_prob_y'])
    df_of_all_orig['pred_y'] = np.where(df_of_all_orig['pred_prob_y'] > threshold[0], 1, 0)

    # calibration curve using the inherent package
    numbe_bins = 6
    prob_true, prob_pred = calibration_curve(df_of_all_orig[task], df_of_all_orig['pred_prob_y'], n_bins=numbe_bins+1)

    # binning and then computing

    sequence_of_scalars = np.linspace(0, 1, numbe_bins+1)
    categorical_object = pd.cut(df_of_all_orig['pred_prob_y'], sequence_of_scalars, right=False)
    # print(categorical_object)
    bin_freq_wrt_mean_pred = pd.value_counts(categorical_object).sort_index().values
    standard_error = prob_true * (1 - prob_true) / bin_freq_wrt_mean_pred  # binomial standard error p*(1-p)/n

    # hist = df_of_all_orig['pred_prob_y'].hist(bins = 20)

    print('Calibration details')
    print('prob true', np.round(prob_true, decimals=4))
    print('prob pred', np.round(prob_pred, decimals=4))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.scatter(prob_pred, prob_true, label="Presented model", color='red', s=50)
    plt.errorbar(prob_pred, prob_true, yerr=standard_error, linestyle='None')  # the error bars are very small
    plt.legend(ncol=4, prop={'size': 10})
    # ax.set_xticklabels(Pred_outcome_rate_to_plot['x_ticks'], ha='right')
    # plt.xticks(Pred_outcome_rate_to_plot['x'], Pred_outcome_rate_to_plot['x_ticks'], rotation = 90)
    # start, end = ax.get_ylim()
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.tick_params(axis='y', which='major', labelsize=10)
    plt.title(" Calibration curve for the whole validation data")
    plt.ylabel(" Fraction of positives")
    plt.xlabel(" Mean predicted value")
    # plt.legend()
    plt.savefig(fig_saving_dir+"/Calibration_curve_final_model_report_card_with_" + str(num_of_bts) + "_BTS.pdf")
    plt.savefig(fig_saving_dir+"/Calibration_curve_final_model_report_card_with_" + str(num_of_bts) + "_BTS.png")
    plt.close()

    # getting the histogram of the predicted risk

    # # getting the histogram of the predicted risk
    # fig, ax = plt.subplots()
    # N, bins, patches = plt.hist(df_of_all_orig['pred_prob_y'], bins=50)
    # plt.setp(
    #     [p for p, b in zip(patches, bins) if b >= np.round(individual_patient_details['predicted_risk'], decimals=3)],
    #     color='r')
    # plt.text(0.2, int(np.max(N)) - 500,
    #          " Bar colour changes (red line) at the current patient's \n risk score which is " + str(
    #              np.round(individual_patient_details['predicted_risk'], decimals=3)))
    # ax.axvline(x=np.round(individual_patient_details['predicted_risk'], decimals=3), color='red')
    # plt.tick_params(axis='x', which='major', labelsize=10)
    # plt.tick_params(axis='y', which='major', labelsize=10)
    # plt.title(" Histogram for predicted risk on the whole validation data")
    # plt.xlabel(" Predicted risk ")
    # plt.ylabel(" Bin frequency")
    # plt.savefig(
    #     fig_saving_dir + "/Histogram_Predicted_risk_final_model_report_card_with_" + str(num_of_bts) + "_BTS.pdf")
    # plt.savefig(
    #     fig_saving_dir + "/Histogram_Predicted_risk_final_model_report_card_with_" + str(num_of_bts) + "_BTS.png")
    # plt.close()

    # # getting age in its original form
    # df_of_all_orig['Age_de'] = age_mean + age_std * df_of_all_orig['age']
    age_sorted_freq = MIMIC_sens_ohe['age'].value_counts().sort_index()
    age_sorted_freq = age_sorted_freq.reset_index()
    age_sorted_freq.rename(columns={"index": "Age", 'age': "Frequency"}, inplace=True)

    # age range partitionaing based on equal patients in each
    temp0 = 0
    for i in range(len(age_sorted_freq)):
        if temp0 < len(df_of_all_orig) / 3:
            temp0 = temp0 + age_sorted_freq.iloc[i]["Frequency"]
            interval1_end = age_sorted_freq.iloc[i]["Age"]
            temp1 = temp0
        elif temp1 < 2 * len(df_of_all_orig) / 3:
            temp1 = temp1 + age_sorted_freq.iloc[i]["Frequency"]
            interval2_end = age_sorted_freq.iloc[i]["Age"]

    # alternative age range partitioning
    # interval1_end = min(df_of_all_orig['Age_de']) + (max(df_of_all_orig['Age_de']) - min(df_of_all_orig['Age_de'])) / 3
    # interval2_end = min(df_of_all_orig['Age_de']) + 2 * (max(df_of_all_orig['Age_de']) - min(df_of_all_orig['Age_de'])) / 3

    # adding some new columns to one hot encode the age variable
    MIMIC_sens_ohe['age_0.0'] = np.where(MIMIC_sens_ohe['age'] < interval1_end, 1, 0)
    MIMIC_sens_ohe['age_1.0'] = np.where(
        (MIMIC_sens_ohe['age'] >= interval1_end) & (MIMIC_sens_ohe['age'] < interval2_end), 1,
        0)
    MIMIC_sens_ohe['age_2.0'] = np.where(MIMIC_sens_ohe['age'] >= interval2_end, 1, 0)

    MIMIC_sens_ohe['postop_del'] = df_of_all_orig['postop_del']

    # # adding new column to one hot encode sex variable for ease of computation ahead
    # df_of_all_orig['Sex_0.0'] = np.where(df_of_all_orig['Sex']==0, 1,0)
    # df_of_all_orig['Sex_1.0'] = np.where(df_of_all_orig['Sex']==1, 1,0)

    # STARTING THE BOOTSTRAP PROCESS HERE

    ACC = np.zeros(num_of_bts)
    AUROC = np.zeros(num_of_bts)
    PPV = np.zeros(num_of_bts)
    Sensitivity = np.zeros(num_of_bts)

    Pred_outcome_rate = np.zeros(num_of_bts)
    Avg_risk_score_list = []

    ACC_level_wise = np.zeros((num_of_bts, 3,
                               3))  # last dimension is for the number of sensitive variables (race, sex, age), second dimension is for the maximum number of level for each sens variable, currently age has 3 levels
    AUROC_level_wise = np.zeros((num_of_bts, 3, 3))
    PPV_level_wise = np.zeros((num_of_bts, 3, 3))
    Sensitivity_level_wise = np.zeros((num_of_bts, 3, 3))

    Pred_outcome_rate_level_wise = np.zeros((num_of_bts, 3, 3))
    # Average_risk_score_level_wise_list = {}

    ACC_level_wise_after_psm = np.zeros((num_of_bts, 3,
                                         6))  # last dimension is all possible combinations of different levels in sensitive variables for psm , second dimension is for the maximum number of level for each sens variable, currently age has 3 levels
    AUROC_level_wise_after_psm = np.zeros((num_of_bts, 3, 6))
    PPV_level_wise_after_psm = np.zeros((num_of_bts, 3, 6))
    Sensitivity_level_wise_after_psm = np.zeros((num_of_bts, 3, 6))

    Pred_outcome_rate_level_wise_after_psm = np.zeros((num_of_bts, 3, 6))
    # Average_risk_score_level_wise_list_after_psm = np.zeros((num_of_bts, 3, 4))

    # mapping of sex need to be checked
    sens_var_level_dict = {'Sex_0': 'Male', 'Sex_1': 'Female', 'RACE_1': 'Black',
                           'RACE_0': 'White', 'RACE_-1': 'Hispanic',
                           'age_0.0': "Age\n [" + str(int(min(MIMIC_sens_ohe['age']))) + " - " + str(
                               int(interval1_end)) + ")",
                           'age_1.0': "Age\n [" + str(int(interval1_end)) + " - " + str(int(interval2_end)) + ")",
                           'age_2.0': "Age\n [" + str(int(interval2_end)) + " - " + str(
                               int(max(MIMIC_sens_ohe['age']))) + "]"}


    # get all the column names and different levels for them
    frequency_level_wise = {}
    frequency_level_wise_pos = {}
    frequency_level_wise_neg = {}
    ohe_sens_var = ['RACE_-1', 'RACE_0', 'RACE_1', 'age_0.0', 'age_1.0', 'age_2.0', 'Sex_0', 'Sex_1']
    level_dict_sens_var = dict(zip(sens_var, np.zeros(len(sens_var))))
    for i in ohe_sens_var:
        for j in sens_var:
            if i.split("_")[0] == j:
                level_dict_sens_var[j] = level_dict_sens_var[j] + 1
                frequency_level_wise[i] = len(MIMIC_sens_ohe[MIMIC_sens_ohe[i] == 1])
                frequency_level_wise_pos[i] = len(
                    MIMIC_sens_ohe[(MIMIC_sens_ohe[i] == 1) & (MIMIC_sens_ohe['postop_del'] == 1)])
                frequency_level_wise_neg[i] = len(
                    MIMIC_sens_ohe[(MIMIC_sens_ohe[i] == 1) & (MIMIC_sens_ohe['postop_del'] == 0)])

    # frequency for pos and negative cases sep
    freq_level_del = pd.DataFrame(dtype=object)
    freq_level_del['Feature_level'] = frequency_level_wise.keys()
    freq_level_del['Patient_freq_Delpos'] = frequency_level_wise_pos.values()
    freq_level_del['Patient_freq_Delneg'] = frequency_level_wise_neg.values()


    # performing propensity matching on the whole data before bootstrapping and getting the feature level distribution after PSM
    frequency_level_wise_psm = {}
    frequency_level_wise_psm_pos = {}
    frequency_level_wise_psm_neg = {}
    for sv in sens_var:
        index_levels = {}
        for name in ohe_sens_var:
            if name.split("_")[0] == sv:
                index_levels[name] = list(MIMIC_sens_ohe.loc[MIMIC_sens_ohe[name] == 1].index)

        if sv == 'RACE':
            te_indixes_BlacktoWhite, te_indixes_BlacktoOthers, te_indixes_WhitetoOthers = Prop_score_comp(
                df_of_all_orig.copy(), index_levels,
                features, MIMIC_sens_ohe)
            data_after_psm_BtoW_full = df_of_all_orig.loc[te_indixes_BlacktoWhite].copy()
            dem_data_after_psm_BtoW_full = MIMIC_sens_ohe.loc[te_indixes_BlacktoWhite].copy()
            for name in ohe_sens_var:
                if name.split("_")[0] == sv:
                    frequency_level_wise_psm[name + 'psm_BW'] = len(
                        list(dem_data_after_psm_BtoW_full.loc[dem_data_after_psm_BtoW_full[name] == 1].index))
                    frequency_level_wise_psm_pos[name + 'psm_BW'] = len(
                        dem_data_after_psm_BtoW_full[(dem_data_after_psm_BtoW_full[name] == 1) & (dem_data_after_psm_BtoW_full['postop_del']==1)])
                    frequency_level_wise_psm_neg[name + 'psm_BW'] = len(
                        dem_data_after_psm_BtoW_full[(dem_data_after_psm_BtoW_full[name] == 1) & (
                                    dem_data_after_psm_BtoW_full['postop_del'] == 0)])

            data_after_psm_OtoB_full = df_of_all_orig.loc[te_indixes_BlacktoOthers].copy()
            dem_data_after_psm_OtoB_full = MIMIC_sens_ohe.loc[te_indixes_BlacktoOthers].copy()
            for name in ohe_sens_var:
                if name.split("_")[0] == sv:
                    frequency_level_wise_psm[name + 'psm_OB'] = len(
                        list(dem_data_after_psm_OtoB_full.loc[dem_data_after_psm_OtoB_full[name] == 1].index))
                    frequency_level_wise_psm_pos[name + 'psm_OB'] = len(
                        dem_data_after_psm_OtoB_full[(dem_data_after_psm_OtoB_full[name] == 1) & (dem_data_after_psm_OtoB_full['postop_del']==1)])
                    frequency_level_wise_psm_neg[name + 'psm_OB'] = len(
                        dem_data_after_psm_OtoB_full[(dem_data_after_psm_OtoB_full[name] == 1) & (
                                    dem_data_after_psm_OtoB_full['postop_del'] == 0)])

            data_after_psm_OtoW_full = df_of_all_orig.loc[te_indixes_WhitetoOthers].copy()
            dem_data_after_psm_OtoW_full = MIMIC_sens_ohe.loc[te_indixes_WhitetoOthers].copy()
            for name in ohe_sens_var:
                if name.split("_")[0] == sv:
                    frequency_level_wise_psm[name + 'psm_OW'] = len(
                        list(dem_data_after_psm_OtoW_full.loc[dem_data_after_psm_OtoW_full[name] == 1].index))
                    frequency_level_wise_psm_pos[name + 'psm_OW'] = len(
                        dem_data_after_psm_OtoW_full[(dem_data_after_psm_OtoW_full[name] == 1) & (dem_data_after_psm_OtoW_full['postop_del']==1)])
                    frequency_level_wise_psm_neg[name + 'psm_OW'] = len(
                        dem_data_after_psm_OtoW_full[(dem_data_after_psm_OtoW_full[name] == 1) & (
                                    dem_data_after_psm_OtoW_full['postop_del'] == 0)])
        if sv == 'age':
            final_te_indixes_1to0, final_te_indixes_1to2 = Prop_score_comp(df_of_all_orig.copy(), index_levels,
                                                                           features, MIMIC_sens_ohe)
            data_after_psm_1to0_full = df_of_all_orig.loc[final_te_indixes_1to0].copy()
            dem_data_after_psm_1to0_full = MIMIC_sens_ohe.loc[final_te_indixes_1to0].copy()
            for name in ohe_sens_var:
                if name.split("_")[0] == sv:
                    frequency_level_wise_psm[name + 'psm10'] = len(
                        list(dem_data_after_psm_1to0_full.loc[dem_data_after_psm_1to0_full[name] == 1].index))
                    frequency_level_wise_psm_pos[name + 'psm10'] = len(
                        dem_data_after_psm_1to0_full[(dem_data_after_psm_1to0_full[name] == 1) & (dem_data_after_psm_1to0_full['postop_del']==1)])
                    frequency_level_wise_psm_neg[name + 'psm10'] = len(
                        dem_data_after_psm_1to0_full[(dem_data_after_psm_1to0_full[name] == 1) & (
                                    dem_data_after_psm_1to0_full['postop_del'] == 0)])
            data_after_psm_1to2_full = df_of_all_orig.loc[final_te_indixes_1to2].copy()
            dem_data_after_psm_1to2_full = MIMIC_sens_ohe.loc[final_te_indixes_1to2].copy()
            for name in ohe_sens_var:
                if name.split("_")[0] == sv:
                    frequency_level_wise_psm[name + 'psm12'] = len(
                        list(dem_data_after_psm_1to2_full.loc[dem_data_after_psm_1to2_full[name] == 1].index))
                    frequency_level_wise_psm_pos[name + 'psm12'] = len(
                        dem_data_after_psm_1to2_full[(dem_data_after_psm_1to2_full[name] == 1) & (dem_data_after_psm_1to2_full['postop_del']==1)])
                    frequency_level_wise_psm_neg[name + 'psm12'] = len(
                        dem_data_after_psm_1to2_full[(dem_data_after_psm_1to2_full[name] == 1) & (
                                    dem_data_after_psm_1to2_full['postop_del'] == 0)])
        if sv == 'Sex':
            features_temp = features
            final_te_indixes = Prop_score_comp(df_of_all_orig.copy(), index_levels, features_temp, MIMIC_sens_ohe)
            data_after_psm_full_sex = df_of_all_orig.loc[final_te_indixes].copy()
            dem_data_after_psm = MIMIC_sens_ohe.loc[final_te_indixes].copy()
            for name in ohe_sens_var:
                if name.split("_")[0] == sv:
                    frequency_level_wise_psm[name] = len(list(dem_data_after_psm.loc[dem_data_after_psm[name] == 1].index))
                    frequency_level_wise_psm_pos[name] = len(dem_data_after_psm[(dem_data_after_psm[name] == 1) & (dem_data_after_psm['postop_del'] == 1)])
                    frequency_level_wise_psm_neg[name] = len(dem_data_after_psm[(dem_data_after_psm[name] == 1) & (dem_data_after_psm['postop_del'] == 0)])

    freq_level_del_psm = pd.DataFrame(dtype=object)
    freq_level_del_psm['Feature_level'] = frequency_level_wise_psm.keys()
    freq_level_del_psm['Patient_freq_Delpos'] = frequency_level_wise_psm_pos.values()
    freq_level_del_psm['Patient_freq_Delneg'] = frequency_level_wise_psm_neg.values()

    counter = 0  # counter for saving the risks across different bootstrap samples
    sens_var_levels = []
    for bts in range(num_of_bts):

        print("\n **********************************************************************")
        print(" Bootstrap sample number  ", bts, )
        print("**********************************************************************\n")

        data_pos = df_of_all_orig[df_of_all_orig[task] == 1]
        data_neg = df_of_all_orig[df_of_all_orig[task] == 0]
        data_pos_bts = data_pos.sample(n=len(data_pos), replace=True, random_state=bts, axis=0)
        data_neg_bts = data_neg.sample(n=len(data_neg), replace=True, random_state=bts, axis=0)

        df_of_all = pd.concat([data_pos_bts, data_neg_bts], axis=0)
        dem_df_of_all = MIMIC_sens_ohe.loc[df_of_all.index].copy()

        Avg_risk_score_list.append(df_of_all['pred_prob_y'].values)
        risk_list = []
        risk_list_afterPsm = []

        # metrics like acc, auroc, PPV and Sensitivity
        Pred_outcome_rate[bts] = len(df_of_all[df_of_all['pred_y'] == 1]) / len(df_of_all)

        ACC[bts] = np.mean(np.where(df_of_all['pred_y'] == df_of_all[task], 1, 0))
        AUROC[bts] = roc_auc_score(df_of_all[task], df_of_all['pred_prob_y'])
        PPV[bts] = len(df_of_all[df_of_all[task] == 1][df_of_all[df_of_all[task] == 1]['pred_y'] == 1]) / len(
            df_of_all[df_of_all['pred_y'] == 1])  # True_positive/total predicted positive
        Sensitivity[bts] = len(
            df_of_all[df_of_all[task] == 1][df_of_all[df_of_all[task] == 1]['pred_y'] == 1]) / len(
            df_of_all[df_of_all[task] == 1])  # True_positive/total true positive

        print("\n ----------------------- Performance on whole validation data ----------------------- ")
        print("acc \t auroc \t PPV \t Sensitivity")
        print(np.round(ACC[bts], decimals=3), "\t", np.round(AUROC[bts], decimals=3), "\t ",
              np.round(PPV[bts], decimals=3), "\t ",
              np.round(Sensitivity[bts], decimals=3), "\n")
        print("----------------------- \n")

        sv_counter = 0
        sv_counter_psm = 0
        for sv in sens_var:
            print("\n **********************************************************************")
            print("Results for sensitive variable ", sv, )
            print("**********************************************************************\n")

            index_levels = {}
            for name in ohe_sens_var:
                if name.split("_")[0] == sv:
                    index_levels[name] = list(dem_df_of_all.loc[dem_df_of_all[name] == 1].index)

            # task rates
            predicted_outcome_rate_level_wise = {}
            risk_level_wise = {}
            true_outcome_rate_level_wise = {}
            # groupwise performance measure
            accuracy_level_wise = {}
            auroc_level_wise = {}
            auprc_level_wise = {}
            ppv_level_wise = {}
            sensitivity_level_wise = {}

            level_counter = 0
            for feat in index_levels.keys():
                if len(index_levels[feat]) > 100:
                    # task rates
                    risk_level_wise[feat] = df_of_all['pred_prob_y'].index.isin(index_levels[feat])
                    risk_list.append(list(df_of_all['pred_prob_y'][df_of_all.index.isin(index_levels[feat])].values))

                    true_outcome_rate_level_wise[feat] = len(df_of_all[df_of_all[task] == 1][
                                                                 df_of_all[df_of_all[task] == 1].index.isin(
                                                                     index_levels[feat])]) / len(index_levels[feat])
                    predicted_outcome_rate_level_wise[feat] = len(df_of_all[df_of_all['pred_y'] == 1][
                                                                      df_of_all[df_of_all['pred_y'] == 1].index.isin(
                                                                          index_levels[feat])]) / len(
                        index_levels[feat])

                    Pred_outcome_rate_level_wise[bts, level_counter, sv_counter] = predicted_outcome_rate_level_wise[
                        feat]

                    # groupwise performance metrics
                    accuracy_level_wise[feat] = np.mean(
                        np.where(
                            df_of_all.loc[index_levels[feat]]['pred_y'] == df_of_all.loc[index_levels[feat]][task],
                            1, 0))
                    if len(df_of_all.loc[index_levels[feat]][task].unique()) == 2:
                        auroc_level_wise[feat] = roc_auc_score(df_of_all.loc[index_levels[feat]][task],
                                                               df_of_all.loc[index_levels[feat]]['pred_prob_y'])
                        auprc_level_wise[feat] = average_precision_score(df_of_all.loc[index_levels[feat]][task],
                                                                         df_of_all.loc[index_levels[feat]][
                                                                             'pred_prob_y'])
                        AUROC_level_wise[bts, level_counter, sv_counter] = auroc_level_wise[feat]

                    if (len(df_of_all.loc[index_levels[feat]][
                                df_of_all.loc[index_levels[feat]]['pred_y'] == 1]) != 0) & (len(
                        df_of_all.loc[index_levels[feat]][df_of_all.loc[index_levels[feat]][task] == 1]) != 0):
                        ppv_level_wise[feat] = len(
                            df_of_all.loc[index_levels[feat]][df_of_all.loc[index_levels[feat]][task] == 1][
                                df_of_all.loc[index_levels[feat]][df_of_all.loc[index_levels[feat]][task] == 1][
                                    'pred_y'] == 1]) / len(
                            df_of_all.loc[index_levels[feat]][df_of_all.loc[index_levels[feat]]['pred_y'] == 1])
                        sensitivity_level_wise[feat] = len(
                            df_of_all.loc[index_levels[feat]][df_of_all.loc[index_levels[feat]][task] == 1][
                                df_of_all.loc[index_levels[feat]][df_of_all.loc[index_levels[feat]][task] == 1][
                                    'pred_y'] == 1]) / len(
                            df_of_all.loc[index_levels[feat]][df_of_all.loc[index_levels[feat]][task] == 1])

                        PPV_level_wise[bts, level_counter, sv_counter] = ppv_level_wise[feat]
                        Sensitivity_level_wise[bts, level_counter, sv_counter] = sensitivity_level_wise[feat]

                    ACC_level_wise[bts, level_counter, sv_counter] = accuracy_level_wise[feat]

                    level_counter = level_counter + 1
                    if bts == 0:
                        sens_var_levels.append(feat)

            print("\n \n Outcome rate dictionary when the sensitive variable is", sv, "\n")
            print("True values ", true_outcome_rate_level_wise)
            print(" Predicted values ", predicted_outcome_rate_level_wise)

            print("\n \n Performance metrics in dictionary form when the sensitive variable is", sv, "\n")
            print(" Accuracy: ", accuracy_level_wise)
            print(" AUROC: ", auroc_level_wise)
            print(" AUPRC: ", auprc_level_wise)
            print(" PPV: ", ppv_level_wise)
            print(" Sensitivity: ", sensitivity_level_wise)

            # getting the propensity matched scores
            if sv == 'age':
                # if bts == 0:
                #     features = features + list(index_levels.keys())
                # final_te_indixes_1to0, final_te_indixes_1to2 = Prop_score_comp(df_of_all.copy(), index_levels, features)
                #
                # data_after_psm_1to0 = df_of_all.loc[final_te_indixes_1to0].copy()

                data_pos = data_after_psm_1to0_full[data_after_psm_1to0_full[task] == 1]
                data_neg = data_after_psm_1to0_full[data_after_psm_1to0_full[task] == 0]
                data_pos_bts = data_pos.sample(n=len(data_pos), replace=True, random_state=bts, axis=0)
                data_neg_bts = data_neg.sample(n=len(data_neg), replace=True, random_state=bts, axis=0)

                data_after_psm_1to0 = pd.concat([data_pos_bts, data_neg_bts], axis=0)
                dem_data_after_psm_1to0 = MIMIC_sens_ohe.loc[data_after_psm_1to0.index]

                index_levels_after_psm_1to0 = {}
                for name in ohe_sens_var:
                    if name.split("_")[0] == sv:
                        index_levels_after_psm_1to0[name] = list(
                            dem_data_after_psm_1to0.loc[dem_data_after_psm_1to0[name] == 1].index)

                # outcome rates
                predicted_outcome_rate_level_wise_after_psm_1to0 = {}
                true_outcome_rate_level_wise_after_psm_1to0 = {}
                risk_level_wise_after_psm_1to0 = {}
                # groupwise performance measure
                accuracy_level_wise_after_psm_1to0 = {}
                auroc_level_wise_after_psm_1to0 = {}
                auprc_level_wise_after_psm_1to0 = {}
                ppv_level_wise_after_psm_1to0 = {}
                sensitivity_level_wise_after_psm_1to0 = {}

                level_counter_after_psm_1to0 = 0
                for feat in index_levels_after_psm_1to0.keys():
                    if len(index_levels_after_psm_1to0[feat]) > 100:
                        # outcome rates

                        risk_level_wise_after_psm_1to0[feat] = data_after_psm_1to0['pred_prob_y'].index.isin(
                            index_levels_after_psm_1to0[
                                feat])
                        risk_list_afterPsm.append(list(data_after_psm_1to0['pred_prob_y'][data_after_psm_1to0.index.isin(
                            index_levels_after_psm_1to0[feat])].values))

                        true_outcome_rate_level_wise_after_psm_1to0[feat] = len(
                            data_after_psm_1to0[data_after_psm_1to0[task] == 1][
                                data_after_psm_1to0[data_after_psm_1to0[
                                                        task] == 1].index.isin(
                                    index_levels_after_psm_1to0[
                                        feat])]) / len(
                            index_levels_after_psm_1to0[feat])
                        predicted_outcome_rate_level_wise_after_psm_1to0[feat] = len(
                            data_after_psm_1to0[data_after_psm_1to0['pred_y'] == 1][
                                data_after_psm_1to0[data_after_psm_1to0['pred_y'] == 1].index.isin(
                                    index_levels_after_psm_1to0[feat])]) / len(
                            index_levels_after_psm_1to0[feat])

                        Pred_outcome_rate_level_wise_after_psm[bts, level_counter_after_psm_1to0, sv_counter_psm] = \
                            predicted_outcome_rate_level_wise_after_psm_1to0[feat]

                        # groupwise performance metrics
                        accuracy_level_wise_after_psm_1to0[feat] = np.mean(
                            np.where(
                                data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]]['pred_y'] ==
                                data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][task],
                                1, 0))
                        if len(data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][task].unique()) == 2:
                            auroc_level_wise_after_psm_1to0[feat] = roc_auc_score(
                                data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][task],
                                data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]]['pred_prob_y'])
                            AUROC_level_wise_after_psm[bts, level_counter_after_psm_1to0, sv_counter_psm] = \
                                auroc_level_wise_after_psm_1to0[feat]

                            auprc_level_wise_after_psm_1to0[feat] = average_precision_score(
                                data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][task],
                                data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]]['pred_prob_y'])
                        if (len(data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][
                                    data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][
                                        'pred_y'] == 1]) != 0) & (len(
                            data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][
                                data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][task] == 1]) != 0):
                            ppv_level_wise_after_psm_1to0[feat] = len(
                                data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][
                                    data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][task] == 1][
                                    data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][
                                        data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][task] == 1][
                                        'pred_y'] == 1]) / len(
                                data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][
                                    data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]]['pred_y'] == 1])
                            sensitivity_level_wise_after_psm_1to0[feat] = len(
                                data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][
                                    data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][task] == 1][
                                    data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][
                                        data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][task] == 1][
                                        'pred_y'] == 1]) / len(
                                data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][
                                    data_after_psm_1to0.loc[index_levels_after_psm_1to0[feat]][task] == 1])
                            PPV_level_wise_after_psm[bts, level_counter_after_psm_1to0, sv_counter_psm] = \
                                ppv_level_wise_after_psm_1to0[
                                    feat]
                            Sensitivity_level_wise_after_psm[bts, level_counter_after_psm_1to0, sv_counter_psm] = \
                                sensitivity_level_wise_after_psm_1to0[feat]
                        ACC_level_wise_after_psm[bts, level_counter_after_psm_1to0, sv_counter_psm] = \
                            accuracy_level_wise_after_psm_1to0[feat]

                        level_counter_after_psm_1to0 = level_counter_after_psm_1to0 + 1

                print("\n \n After Propensity score matching: Outcome rate dictionary when the sensitive variable is",
                      sv, "\n")
                print("True values ", true_outcome_rate_level_wise_after_psm_1to0)
                print(" Predicted values ", predicted_outcome_rate_level_wise_after_psm_1to0)

                print(
                    "\n \n After Propensity score matching: Performance metrics in dictionary form when the sensitive variable is",
                    sv, "\n")
                print(" Accuracy: ", accuracy_level_wise_after_psm_1to0)
                print(" AUROC: ", auroc_level_wise_after_psm_1to0)
                print(" AUPRC: ", auprc_level_wise_after_psm_1to0)
                print(" PPV: ", ppv_level_wise_after_psm_1to0)
                print(" Sensitivity: ", sensitivity_level_wise_after_psm_1to0)

                sv_counter_psm = sv_counter_psm +1
                ############################################################

                # data_after_psm_1to2 = df_of_all.loc[final_te_indixes_1to2].copy()

                data_pos = data_after_psm_1to2_full[data_after_psm_1to2_full[task] == 1]
                data_neg = data_after_psm_1to2_full[data_after_psm_1to2_full[task] == 0]
                data_pos_bts = data_pos.sample(n=len(data_pos), replace=True, random_state=bts, axis=0)
                data_neg_bts = data_neg.sample(n=len(data_neg), replace=True, random_state=bts, axis=0)

                data_after_psm_1to2 = pd.concat([data_pos_bts, data_neg_bts], axis=0)
                dem_data_after_psm_1to2 = MIMIC_sens_ohe.loc[data_after_psm_1to2.index]

                index_levels_after_psm_1to2 = {}
                for name in ohe_sens_var:
                    if name.split("_")[0] == sv:
                        index_levels_after_psm_1to2[name] = list(
                            dem_data_after_psm_1to2.loc[dem_data_after_psm_1to2[name] == 1].index)

                # outcome rates
                predicted_outcome_rate_level_wise_after_psm_1to2 = {}
                true_outcome_rate_level_wise_after_psm_1to2 = {}
                risk_level_wise_after_psm_1to2 = {}
                # groupwise performance measure
                accuracy_level_wise_after_psm_1to2 = {}
                auroc_level_wise_after_psm_1to2 = {}
                auprc_level_wise_after_psm_1to2 = {}
                ppv_level_wise_after_psm_1to2 = {}
                sensitivity_level_wise_after_psm_1to2 = {}

                level_counter_after_psm_1to2 = 1
                for feat in index_levels_after_psm_1to2.keys():
                    if len(index_levels_after_psm_1to2[feat]) > 100:
                        # outcome rates

                        risk_level_wise_after_psm_1to2[feat] = data_after_psm_1to2['pred_prob_y'].index.isin(
                            index_levels_after_psm_1to2[
                                feat])
                        risk_list_afterPsm.append(list(data_after_psm_1to2['pred_prob_y'][data_after_psm_1to2.index.isin(
                            index_levels_after_psm_1to2[feat])].values))

                        true_outcome_rate_level_wise_after_psm_1to2[feat] = len(
                            data_after_psm_1to2[data_after_psm_1to2[task] == 1][
                                data_after_psm_1to2[data_after_psm_1to2[
                                                        task] == 1].index.isin(
                                    index_levels_after_psm_1to2[
                                        feat])]) / len(
                            index_levels_after_psm_1to2[feat])
                        predicted_outcome_rate_level_wise_after_psm_1to2[feat] = len(
                            data_after_psm_1to2[data_after_psm_1to2['pred_y'] == 1][
                                data_after_psm_1to2[data_after_psm_1to2['pred_y'] == 1].index.isin(
                                    index_levels_after_psm_1to2[feat])]) / len(
                            index_levels_after_psm_1to2[feat])

                        Pred_outcome_rate_level_wise_after_psm[bts, level_counter_after_psm_1to2, sv_counter_psm] = \
                            predicted_outcome_rate_level_wise_after_psm_1to2[feat]

                        # groupwise performance metrics
                        accuracy_level_wise_after_psm_1to2[feat] = np.mean(
                            np.where(
                                data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]]['pred_y'] ==
                                data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][task],
                                1, 0))
                        if len(data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][task].unique()) == 2:
                            auroc_level_wise_after_psm_1to2[feat] = roc_auc_score(
                                data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][task],
                                data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]]['pred_prob_y'])
                            AUROC_level_wise_after_psm[bts, level_counter_after_psm_1to2, sv_counter_psm] = \
                                auroc_level_wise_after_psm_1to2[feat]

                            auprc_level_wise_after_psm_1to2[feat] = average_precision_score(
                                data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][task],
                                data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]]['pred_prob_y'])
                        if (len(data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][
                                    data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][
                                        'pred_y'] == 1]) != 0) & (len(
                            data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][
                                data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][task] == 1]) != 0):
                            ppv_level_wise_after_psm_1to2[feat] = len(
                                data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][
                                    data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][task] == 1][
                                    data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][
                                        data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][task] == 1][
                                        'pred_y'] == 1]) / len(
                                data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][
                                    data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]]['pred_y'] == 1])
                            sensitivity_level_wise_after_psm_1to2[feat] = len(
                                data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][
                                    data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][task] == 1][
                                    data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][
                                        data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][task] == 1][
                                        'pred_y'] == 1]) / len(
                                data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][
                                    data_after_psm_1to2.loc[index_levels_after_psm_1to2[feat]][task] == 1])
                            PPV_level_wise_after_psm[bts, level_counter_after_psm_1to2, sv_counter_psm] = \
                                ppv_level_wise_after_psm_1to2[
                                    feat]
                            Sensitivity_level_wise_after_psm[bts, level_counter_after_psm_1to2, sv_counter_psm] = \
                                sensitivity_level_wise_after_psm_1to2[feat]
                        ACC_level_wise_after_psm[bts, level_counter_after_psm_1to2, sv_counter_psm] = \
                            accuracy_level_wise_after_psm_1to2[feat]

                        level_counter_after_psm_1to2 = level_counter_after_psm_1to2 + 1

                print("\n \n After Propensity score matching: Outcome rate dictionary when the sensitive variable is",
                      sv, "\n")
                print("True values ", true_outcome_rate_level_wise_after_psm_1to2)
                print(" Predicted values ", predicted_outcome_rate_level_wise_after_psm_1to2)

                print(
                    "\n \n After Propensity score matching: Performance metrics in dictionary form when the sensitive variable is",
                    sv, "\n")
                print(" Accuracy: ", accuracy_level_wise_after_psm_1to2)
                print(" AUROC: ", auroc_level_wise_after_psm_1to2)
                print(" AUPRC: ", auprc_level_wise_after_psm_1to2)
                print(" PPV: ", ppv_level_wise_after_psm_1to2)
                print(" Sensitivity: ", sensitivity_level_wise_after_psm_1to2)

                sv_counter_psm = sv_counter_psm +1
                print('meh')

            if sv == 'RACE':
                # te_indixes_BlacktoWhite, te_indixes_BlacktoOthers, te_indixes_WhitetoOthers = Prop_score_comp(
                #     df_of_all.copy(), index_levels,
                #     features)
                #
                # data_after_psm_BtoW = df_of_all.loc[te_indixes_BlacktoWhite].copy()

                data_pos = data_after_psm_BtoW_full[data_after_psm_BtoW_full[task] == 1]
                data_neg = data_after_psm_BtoW_full[data_after_psm_BtoW_full[task] == 0]
                data_pos_bts = data_pos.sample(n=len(data_pos), replace=True, random_state=bts, axis=0)
                data_neg_bts = data_neg.sample(n=len(data_neg), replace=True, random_state=bts, axis=0)

                data_after_psm_BtoW = pd.concat([data_pos_bts, data_neg_bts], axis=0)
                dem_data_after_psm_BtoW = MIMIC_sens_ohe.loc[data_after_psm_BtoW.index]

                index_levels_after_psm_BtoW = {}
                for name in ohe_sens_var:
                    if name.split("_")[0] == sv:
                        index_levels_after_psm_BtoW[name] = list(
                            dem_data_after_psm_BtoW.loc[dem_data_after_psm_BtoW[name] == 1].index)

                # outcome rates
                predicted_outcome_rate_level_wise_after_psm_BtoW = {}
                true_outcome_rate_level_wise_after_psm_BtoW = {}
                risk_level_wise_after_psm_BtoW = {}
                # groupwise performance measure
                accuracy_level_wise_after_psm_BtoW = {}
                auroc_level_wise_after_psm_BtoW = {}
                auprc_level_wise_after_psm_BtoW = {}
                ppv_level_wise_after_psm_BtoW = {}
                sensitivity_level_wise_after_psm_BtoW = {}

                level_counter_after_psm_BtoW = 0
                for feat in index_levels_after_psm_BtoW.keys():
                    if len(index_levels_after_psm_BtoW[feat]) > 100:
                        # outcome rates

                        risk_level_wise_after_psm_BtoW[feat] = data_after_psm_BtoW['pred_prob_y'].index.isin(
                            index_levels_after_psm_BtoW[
                                feat])

                        risk_list_afterPsm.append(list(data_after_psm_BtoW['pred_prob_y'][data_after_psm_BtoW.index.isin(
                            index_levels_after_psm_BtoW[feat])].values))

                        true_outcome_rate_level_wise_after_psm_BtoW[feat] = len(
                            data_after_psm_BtoW[data_after_psm_BtoW[task] == 1][
                                data_after_psm_BtoW[data_after_psm_BtoW[
                                                        task] == 1].index.isin(
                                    index_levels_after_psm_BtoW[
                                        feat])]) / len(
                            index_levels_after_psm_BtoW[feat])
                        predicted_outcome_rate_level_wise_after_psm_BtoW[feat] = len(
                            data_after_psm_BtoW[data_after_psm_BtoW['pred_y'] == 1][
                                data_after_psm_BtoW[data_after_psm_BtoW['pred_y'] == 1].index.isin(
                                    index_levels_after_psm_BtoW[feat])]) / len(
                            index_levels_after_psm_BtoW[feat])

                        Pred_outcome_rate_level_wise_after_psm[bts, level_counter_after_psm_BtoW, sv_counter_psm] = \
                            predicted_outcome_rate_level_wise_after_psm_BtoW[feat]

                        # groupwise performance metrics
                        accuracy_level_wise_after_psm_BtoW[feat] = np.mean(
                            np.where(
                                data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]]['pred_y'] ==
                                data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][task],
                                1, 0))
                        if len(data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][task].unique()) == 2:
                            auroc_level_wise_after_psm_BtoW[feat] = roc_auc_score(
                                data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][task],
                                data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]]['pred_prob_y'])
                            AUROC_level_wise_after_psm[bts, level_counter_after_psm_BtoW, sv_counter_psm] = \
                                auroc_level_wise_after_psm_BtoW[feat]

                            auprc_level_wise_after_psm_BtoW[feat] = average_precision_score(
                                data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][task],
                                data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]]['pred_prob_y'])
                        if (len(data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][
                                    data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][
                                        'pred_y'] == 1]) != 0) & (len(
                            data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][
                                data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][task] == 1]) != 0):
                            ppv_level_wise_after_psm_BtoW[feat] = len(
                                data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][
                                    data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][task] == 1][
                                    data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][
                                        data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][task] == 1][
                                        'pred_y'] == 1]) / len(
                                data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][
                                    data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]]['pred_y'] == 1])
                            sensitivity_level_wise_after_psm_BtoW[feat] = len(
                                data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][
                                    data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][task] == 1][
                                    data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][
                                        data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][task] == 1][
                                        'pred_y'] == 1]) / len(
                                data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][
                                    data_after_psm_BtoW.loc[index_levels_after_psm_BtoW[feat]][task] == 1])
                            PPV_level_wise_after_psm[bts, level_counter_after_psm_BtoW, sv_counter_psm] = \
                                ppv_level_wise_after_psm_BtoW[
                                    feat]
                            Sensitivity_level_wise_after_psm[bts, level_counter_after_psm_BtoW, sv_counter_psm] = \
                                sensitivity_level_wise_after_psm_BtoW[feat]
                        ACC_level_wise_after_psm[bts, level_counter_after_psm_BtoW, sv_counter_psm] = \
                            accuracy_level_wise_after_psm_BtoW[feat]

                        level_counter_after_psm_BtoW = level_counter_after_psm_BtoW + 1

                print("\n \n After Propensity score matching: Outcome rate dictionary when the sensitive variable is",
                      sv, "\n")
                print("True values ", true_outcome_rate_level_wise_after_psm_BtoW)
                print(" Predicted values ", predicted_outcome_rate_level_wise_after_psm_BtoW)

                print(
                    "\n \n After Propensity score matching: Performance metrics in dictionary form when the sensitive variable is",
                    sv, "\n")
                print(" Accuracy: ", accuracy_level_wise_after_psm_BtoW)
                print(" AUROC: ", auroc_level_wise_after_psm_BtoW)
                print(" AUPRC: ", auprc_level_wise_after_psm_BtoW)
                print(" PPV: ", ppv_level_wise_after_psm_BtoW)
                print(" Sensitivity: ", sensitivity_level_wise_after_psm_BtoW)

                sv_counter_psm = sv_counter_psm + 1
                ############################################################
                # data_after_psm_OtoB = df_of_all.loc[te_indixes_BlacktoOthers].copy()

                data_pos = data_after_psm_OtoB_full[data_after_psm_OtoB_full[task] == 1]
                data_neg = data_after_psm_OtoB_full[data_after_psm_OtoB_full[task] == 0]
                data_pos_bts = data_pos.sample(n=len(data_pos), replace=True, random_state=bts, axis=0)
                data_neg_bts = data_neg.sample(n=len(data_neg), replace=True, random_state=bts, axis=0)

                data_after_psm_OtoB = pd.concat([data_pos_bts, data_neg_bts], axis=0)
                dem_data_after_psm_OtoB = MIMIC_sens_ohe.loc[data_after_psm_OtoB.index]

                index_levels_after_psm_OtoB = {}
                for name in ohe_sens_var:
                    if name.split("_")[0] == sv:
                        index_levels_after_psm_OtoB[name] = list(
                            dem_data_after_psm_OtoB.loc[dem_data_after_psm_OtoB[name] == 1].index)

                predicted_outcome_rate_level_wise_after_psm_OtoB = {}
                true_outcome_rate_level_wise_after_psm_OtoB = {}
                risk_level_wise_after_psm_OtoB = {}
                # groupwise performance measure
                accuracy_level_wise_after_psm_OtoB = {}
                auroc_level_wise_after_psm_OtoB = {}
                auprc_level_wise_after_psm_OtoB = {}
                ppv_level_wise_after_psm_OtoB = {}
                sensitivity_level_wise_after_psm_OtoB = {}

                level_counter_after_psm_OtoB = 0
                for feat in index_levels_after_psm_OtoB.keys():
                    if len(index_levels_after_psm_OtoB[feat]) > 100:
                        # outcome rates

                        risk_level_wise_after_psm_OtoB[feat] = data_after_psm_OtoB['pred_prob_y'].index.isin(
                            index_levels_after_psm_OtoB[
                                feat])

                        risk_list_afterPsm.append(list(data_after_psm_OtoB['pred_prob_y'][data_after_psm_OtoB.index.isin(
                            index_levels_after_psm_OtoB[feat])].values))

                        true_outcome_rate_level_wise_after_psm_OtoB[feat] = len(
                            data_after_psm_OtoB[data_after_psm_OtoB[task] == 1][
                                data_after_psm_OtoB[data_after_psm_OtoB[
                                                        task] == 1].index.isin(
                                    index_levels_after_psm_OtoB[
                                        feat])]) / len(
                            index_levels_after_psm_OtoB[feat])
                        predicted_outcome_rate_level_wise_after_psm_OtoB[feat] = len(
                            data_after_psm_OtoB[data_after_psm_OtoB['pred_y'] == 1][
                                data_after_psm_OtoB[data_after_psm_OtoB['pred_y'] == 1].index.isin(
                                    index_levels_after_psm_OtoB[feat])]) / len(
                            index_levels_after_psm_OtoB[feat])

                        Pred_outcome_rate_level_wise_after_psm[bts, level_counter_after_psm_OtoB, sv_counter_psm] = \
                            predicted_outcome_rate_level_wise_after_psm_OtoB[feat]

                        # groupwise performance metrics
                        accuracy_level_wise_after_psm_OtoB[feat] = np.mean(
                            np.where(
                                data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]]['pred_y'] ==
                                data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][task],
                                1, 0))
                        if len(data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][task].unique()) == 2:
                            auroc_level_wise_after_psm_OtoB[feat] = roc_auc_score(
                                data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][task],
                                data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]]['pred_prob_y'])
                            AUROC_level_wise_after_psm[bts, level_counter_after_psm_OtoB, sv_counter_psm] = \
                                auroc_level_wise_after_psm_OtoB[feat]

                            auprc_level_wise_after_psm_OtoB[feat] = average_precision_score(
                                data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][task],
                                data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]]['pred_prob_y'])
                        if (len(data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][
                                    data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][
                                        'pred_y'] == 1]) != 0) & (len(
                            data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][
                                data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][task] == 1]) != 0):
                            ppv_level_wise_after_psm_OtoB[feat] = len(
                                data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][
                                    data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][task] == 1][
                                    data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][
                                        data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][task] == 1][
                                        'pred_y'] == 1]) / len(
                                data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][
                                    data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]]['pred_y'] == 1])
                            sensitivity_level_wise_after_psm_OtoB[feat] = len(
                                data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][
                                    data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][task] == 1][
                                    data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][
                                        data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][task] == 1][
                                        'pred_y'] == 1]) / len(
                                data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][
                                    data_after_psm_OtoB.loc[index_levels_after_psm_OtoB[feat]][task] == 1])
                            PPV_level_wise_after_psm[bts, level_counter_after_psm_OtoB, sv_counter_psm] = \
                                ppv_level_wise_after_psm_OtoB[
                                    feat]
                            Sensitivity_level_wise_after_psm[bts, level_counter_after_psm_OtoB, sv_counter_psm] = \
                                sensitivity_level_wise_after_psm_OtoB[feat]
                        ACC_level_wise_after_psm[bts, level_counter_after_psm_OtoB, sv_counter_psm] = \
                            accuracy_level_wise_after_psm_OtoB[feat]

                        level_counter_after_psm_OtoB = level_counter_after_psm_OtoB + 2

                print("\n \n After Propensity score matching: Outcome rate dictionary when the sensitive variable is",
                      sv, "\n")
                print("True values ", true_outcome_rate_level_wise_after_psm_OtoB)
                print(" Predicted values ", predicted_outcome_rate_level_wise_after_psm_OtoB)

                print(
                    "\n \n After Propensity score matching: Performance metrics in dictionary form when the sensitive variable is",
                    sv, "\n")
                print(" Accuracy: ", accuracy_level_wise_after_psm_OtoB)
                print(" AUROC: ", auroc_level_wise_after_psm_OtoB)
                print(" AUPRC: ", auprc_level_wise_after_psm_OtoB)
                print(" PPV: ", ppv_level_wise_after_psm_OtoB)
                print(" Sensitivity: ", sensitivity_level_wise_after_psm_OtoB)

                sv_counter_psm = sv_counter_psm +1
                ###########################################################################
                # data_after_psm_OtoW = df_of_all.loc[te_indixes_WhitetoOthers].copy()

                data_pos = data_after_psm_OtoW_full[data_after_psm_OtoW_full[task] == 1]
                data_neg = data_after_psm_OtoW_full[data_after_psm_OtoW_full[task] == 0]
                data_pos_bts = data_pos.sample(n=len(data_pos), replace=True, random_state=bts, axis=0)
                data_neg_bts = data_neg.sample(n=len(data_neg), replace=True, random_state=bts, axis=0)

                data_after_psm_OtoW = pd.concat([data_pos_bts, data_neg_bts], axis=0)
                dem_data_after_psm_OtoW = MIMIC_sens_ohe.loc[data_after_psm_OtoW.index]

                index_levels_after_psm_OtoW = {}
                for name in ohe_sens_var:
                    if name.split("_")[0] == sv:
                        index_levels_after_psm_OtoW[name] = list(
                            dem_data_after_psm_OtoW.loc[dem_data_after_psm_OtoW[name] == 1].index)

                predicted_outcome_rate_level_wise_after_psm_OtoW = {}
                true_outcome_rate_level_wise_after_psm_OtoW = {}
                risk_level_wise_after_psm_OtoW = {}
                # groupwise performance measure
                accuracy_level_wise_after_psm_OtoW = {}
                auroc_level_wise_after_psm_OtoW = {}
                auprc_level_wise_after_psm_OtoW = {}
                ppv_level_wise_after_psm_OtoW = {}
                sensitivity_level_wise_after_psm_OtoW = {}

                level_counter_after_psm_OtoW = 1
                for feat in index_levels_after_psm_OtoW.keys():
                    if len(index_levels_after_psm_OtoW[feat]) > 100:
                        # outcome rates

                        risk_level_wise_after_psm_OtoW[feat] = data_after_psm_OtoW['pred_prob_y'].index.isin(
                            index_levels_after_psm_OtoW[
                                feat])

                        risk_list_afterPsm.append(list(data_after_psm_OtoW['pred_prob_y'][data_after_psm_OtoW.index.isin(
                            index_levels_after_psm_OtoW[feat])].values))

                        true_outcome_rate_level_wise_after_psm_OtoW[feat] = len(
                            data_after_psm_OtoW[data_after_psm_OtoW[task] == 1][
                                data_after_psm_OtoW[data_after_psm_OtoW[
                                                        task] == 1].index.isin(
                                    index_levels_after_psm_OtoW[
                                        feat])]) / len(
                            index_levels_after_psm_OtoW[feat])
                        predicted_outcome_rate_level_wise_after_psm_OtoW[feat] = len(
                            data_after_psm_OtoW[data_after_psm_OtoW['pred_y'] == 1][
                                data_after_psm_OtoW[data_after_psm_OtoW['pred_y'] == 1].index.isin(
                                    index_levels_after_psm_OtoW[feat])]) / len(
                            index_levels_after_psm_OtoW[feat])

                        Pred_outcome_rate_level_wise_after_psm[bts, level_counter_after_psm_OtoW, sv_counter_psm] = \
                            predicted_outcome_rate_level_wise_after_psm_OtoW[feat]

                        # groupwise performance metrics
                        accuracy_level_wise_after_psm_OtoW[feat] = np.mean(
                            np.where(
                                data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]]['pred_y'] ==
                                data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][task],
                                1, 0))
                        if len(data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][task].unique()) == 2:
                            auroc_level_wise_after_psm_OtoW[feat] = roc_auc_score(
                                data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][task],
                                data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]]['pred_prob_y'])
                            AUROC_level_wise_after_psm[bts, level_counter_after_psm_OtoW, sv_counter_psm] = \
                                auroc_level_wise_after_psm_OtoW[feat]

                            auprc_level_wise_after_psm_OtoW[feat] = average_precision_score(
                                data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][task],
                                data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]]['pred_prob_y'])
                        if (len(data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][
                                    data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][
                                        'pred_y'] == 1]) != 0) & (len(
                            data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][
                                data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][task] == 1]) != 0):
                            ppv_level_wise_after_psm_OtoW[feat] = len(
                                data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][
                                    data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][task] == 1][
                                    data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][
                                        data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][task] == 1][
                                        'pred_y'] == 1]) / len(
                                data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][
                                    data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]]['pred_y'] == 1])
                            sensitivity_level_wise_after_psm_OtoW[feat] = len(
                                data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][
                                    data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][task] == 1][
                                    data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][
                                        data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][task] == 1][
                                        'pred_y'] == 1]) / len(
                                data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][
                                    data_after_psm_OtoW.loc[index_levels_after_psm_OtoW[feat]][task] == 1])
                            PPV_level_wise_after_psm[bts, level_counter_after_psm_OtoW, sv_counter_psm] = \
                                ppv_level_wise_after_psm_OtoW[
                                    feat]
                            Sensitivity_level_wise_after_psm[bts, level_counter_after_psm_OtoW, sv_counter_psm] = \
                                sensitivity_level_wise_after_psm_OtoW[feat]
                        ACC_level_wise_after_psm[bts, level_counter_after_psm_OtoW, sv_counter_psm] = \
                            accuracy_level_wise_after_psm_OtoW[feat]

                        level_counter_after_psm_OtoW = level_counter_after_psm_OtoW + 1

                print("\n \n After Propensity score matching: Outcome rate dictionary when the sensitive variable is",
                      sv, "\n")
                print("True values ", true_outcome_rate_level_wise_after_psm_OtoW)
                print(" Predicted values ", predicted_outcome_rate_level_wise_after_psm_OtoW)

                print(
                    "\n \n After Propensity score matching: Performance metrics in dictionary form when the sensitive variable is",
                    sv, "\n")
                print(" Accuracy: ", accuracy_level_wise_after_psm_OtoW)
                print(" AUROC: ", auroc_level_wise_after_psm_OtoW)
                print(" AUPRC: ", auprc_level_wise_after_psm_OtoW)
                print(" PPV: ", ppv_level_wise_after_psm_OtoW)
                print(" Sensitivity: ", sensitivity_level_wise_after_psm_OtoW)

                sv_counter_psm = sv_counter_psm +1

            if sv == 'Sex':
                # final_te_indixes = Prop_score_comp(df_of_all.copy(), index_levels, features+list(index_levels.keys()))
                # data_after_psm = df_of_all.loc[final_te_indixes].copy()

                data_pos = data_after_psm_full_sex[data_after_psm_full_sex[task] == 1]
                data_neg = data_after_psm_full_sex[data_after_psm_full_sex[task] == 0]
                data_pos_bts = data_pos.sample(n=len(data_pos), replace=True, random_state=bts, axis=0)
                data_neg_bts = data_neg.sample(n=len(data_neg), replace=True, random_state=bts, axis=0)

                data_after_psm = pd.concat([data_pos_bts, data_neg_bts], axis=0)
                dem_data_after_psm = MIMIC_sens_ohe.loc[data_after_psm.index]
                index_levels_after_psm = {}
                for name in ohe_sens_var:
                    if name.split("_")[0] == sv:
                        index_levels_after_psm[name] = list(dem_data_after_psm.loc[dem_data_after_psm[name] == 1].index)

                # outcome rates
                predicted_outcome_rate_level_wise_after_psm = {}
                true_outcome_rate_level_wise_after_psm = {}
                risk_level_wise_after_psm = {}
                # groupwise performance measure
                accuracy_level_wise_after_psm = {}
                auroc_level_wise_after_psm = {}
                auprc_level_wise_after_psm = {}
                ppv_level_wise_after_psm = {}
                sensitivity_level_wise_after_psm = {}

                level_counter_after_psm = 0
                for feat in index_levels_after_psm.keys():
                    if len(index_levels_after_psm[feat]) > 100:
                        # outcome rates

                        risk_level_wise_after_psm[feat] = data_after_psm['pred_prob_y'].index.isin(
                            index_levels_after_psm[
                                feat])
                        risk_list_afterPsm.append(list(data_after_psm['pred_prob_y'][data_after_psm.index.isin(
                            index_levels_after_psm[feat])].values))

                        true_outcome_rate_level_wise_after_psm[feat] = len(data_after_psm[data_after_psm[task] == 1][
                                                                               data_after_psm[data_after_psm[
                                                                                                  task] == 1].index.isin(
                                                                                   index_levels_after_psm[
                                                                                       feat])]) / len(
                            index_levels_after_psm[feat])
                        predicted_outcome_rate_level_wise_after_psm[feat] = len(
                            data_after_psm[data_after_psm['pred_y'] == 1][
                                data_after_psm[data_after_psm['pred_y'] == 1].index.isin(
                                    index_levels_after_psm[feat])]) / len(
                            index_levels_after_psm[feat])

                        Pred_outcome_rate_level_wise_after_psm[bts, level_counter_after_psm, sv_counter_psm] = \
                            predicted_outcome_rate_level_wise_after_psm[feat]

                        # groupwise performance metrics
                        accuracy_level_wise_after_psm[feat] = np.mean(
                            np.where(
                                data_after_psm.loc[index_levels_after_psm[feat]]['pred_y'] ==
                                data_after_psm.loc[index_levels_after_psm[feat]][task],
                                1, 0))
                        if len(data_after_psm.loc[index_levels_after_psm[feat]][task].unique()) == 2:
                            auroc_level_wise_after_psm[feat] = roc_auc_score(
                                data_after_psm.loc[index_levels_after_psm[feat]][task],
                                data_after_psm.loc[index_levels_after_psm[feat]]['pred_prob_y'])
                            AUROC_level_wise_after_psm[bts, level_counter_after_psm, sv_counter_psm] = \
                                auroc_level_wise_after_psm[feat]

                            auprc_level_wise_after_psm[feat] = average_precision_score(
                                data_after_psm.loc[index_levels_after_psm[feat]][task],
                                data_after_psm.loc[index_levels_after_psm[feat]]['pred_prob_y'])
                        if (len(data_after_psm.loc[index_levels_after_psm[feat]][
                                    data_after_psm.loc[index_levels_after_psm[feat]]['pred_y'] == 1]) != 0) & (len(
                            data_after_psm.loc[index_levels_after_psm[feat]][
                                data_after_psm.loc[index_levels_after_psm[feat]][task] == 1]) != 0):
                            ppv_level_wise_after_psm[feat] = len(
                                data_after_psm.loc[index_levels_after_psm[feat]][
                                    data_after_psm.loc[index_levels_after_psm[feat]][task] == 1][
                                    data_after_psm.loc[index_levels_after_psm[feat]][
                                        data_after_psm.loc[index_levels_after_psm[feat]][task] == 1][
                                        'pred_y'] == 1]) / len(
                                data_after_psm.loc[index_levels_after_psm[feat]][
                                    data_after_psm.loc[index_levels_after_psm[feat]]['pred_y'] == 1])
                            sensitivity_level_wise_after_psm[feat] = len(
                                data_after_psm.loc[index_levels_after_psm[feat]][
                                    data_after_psm.loc[index_levels_after_psm[feat]][task] == 1][
                                    data_after_psm.loc[index_levels_after_psm[feat]][
                                        data_after_psm.loc[index_levels_after_psm[feat]][task] == 1][
                                        'pred_y'] == 1]) / len(
                                data_after_psm.loc[index_levels_after_psm[feat]][
                                    data_after_psm.loc[index_levels_after_psm[feat]][task] == 1])
                            PPV_level_wise_after_psm[bts, level_counter_after_psm, sv_counter_psm] = \
                                ppv_level_wise_after_psm[
                                    feat]
                            Sensitivity_level_wise_after_psm[bts, level_counter_after_psm, sv_counter_psm] = \
                                sensitivity_level_wise_after_psm[feat]
                        ACC_level_wise_after_psm[bts, level_counter_after_psm, sv_counter_psm] = \
                            accuracy_level_wise_after_psm[feat]

                        level_counter_after_psm = level_counter_after_psm + 1

                print("\n \n After Propensity score matching: Outcome rate dictionary when the sensitive variable is",
                      sv, "\n")
                print("True values ", true_outcome_rate_level_wise_after_psm)
                print(" Predicted values ", predicted_outcome_rate_level_wise_after_psm)

                print(
                    "\n \n After Propensity score matching: Performance metrics in dictionary form when the sensitive variable is",
                    sv, "\n")
                print(" Accuracy: ", accuracy_level_wise_after_psm)
                print(" AUROC: ", auroc_level_wise_after_psm)
                print(" AUPRC: ", auprc_level_wise_after_psm)
                print(" PPV: ", ppv_level_wise_after_psm)
                print(" Sensitivity: ", sensitivity_level_wise_after_psm)

                sv_counter_psm = sv_counter_psm + 1

                print("meh^2")

            sv_counter = sv_counter + 1

        # appending the risk lists to the
        if bts == 0:
            Average_risk_score_level_wise_list = dict(zip(sens_var_levels, risk_list))

            # make a copy of the freq after psm array because it has all the levels of psm
            temp_freq_psm_copy = frequency_level_wise_psm.copy()
            # delete elements from a dictionary that do not have any elements
            to_del_keys = []
            for i in temp_freq_psm_copy.keys():
                if temp_freq_psm_copy[i] <100:
                    to_del_keys.append(i)
            for d in to_del_keys:
                del temp_freq_psm_copy[d]
            Average_risk_score_level_wise_list_after_psm = dict(zip(temp_freq_psm_copy.keys(), risk_list_afterPsm))
        else:
            counter_list = 0
            for levelname in Average_risk_score_level_wise_list.keys():
                Average_risk_score_level_wise_list[levelname].extend(risk_list[counter_list])
                counter_list = counter_list + 1
            counter_list_psm = 0
            for levelname in Average_risk_score_level_wise_list_after_psm.keys():
                Average_risk_score_level_wise_list_after_psm[levelname].extend(risk_list_afterPsm[counter_list_psm])
                counter_list_psm = counter_list_psm + 1
        del df_of_all

    print("'meh^3")

    """ Outcome  rate plots """

    # getting the number of levels for each sensitive variable
    sens_level_count = []
    for i in sens_var:
        counter = 0
        for j in range(len(sens_var_levels)):
            if sens_var_levels[j].split("_")[0] == i:
                counter = counter + 1
        sens_level_count.append(counter)
    #
    # markers = ["o", "^", "*"]
    # texts = ["Basic", "PSM", "PSM"]
    # patches = [plt.plot([], [], marker=markers[i], ms=10, ls="", mec='k', color='w',
    #                     label="{:s}".format(texts[i]))[0] for i in range(len(texts))]
    #
    # ##########################################  Predicted risk plot ###############################################
    # Mean_Pred_risk_list = [np.round(np.mean(np.array(Average_risk_score_level_wise_list[i])), decimals=4) for i in
    #                        Average_risk_score_level_wise_list.keys()]
    # STD_Pred_risk_list = [np.round(np.std(np.array(Average_risk_score_level_wise_list[i])), decimals=4) for i in
    #                       Average_risk_score_level_wise_list.keys()]
    # Pred_risk_to_plot = pd.DataFrame()
    # Pred_risk_to_plot['Mean_values'] = Mean_Pred_risk_list
    # Pred_risk_to_plot['Std_values'] = STD_Pred_risk_list
    # Pred_risk_to_plot['method'] = " "
    # Pred_risk_to_plot['sens_Var_level'] = " "
    # Pred_risk_to_plot['x_ticks'] = " "
    # idx = 0
    # for j in range(len(sens_var_levels)):
    #     Pred_risk_to_plot.at[idx, 'method'] = "Basic"
    #     Pred_risk_to_plot.at[idx, 'sens_Var_level'] = sens_var_levels[j]
    #     Pred_risk_to_plot.at[idx, 'x_ticks'] = sens_var_level_dict[sens_var_levels[j]]
    #     idx = idx + 1
    # Pred_risk_to_plot['x'] = np.arange(len(Pred_risk_to_plot))
    # mkr_dict = {'Basic': 'o', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'RACE': 'b', 'age': 'g', 'All': 'orange'}
    # groups = Pred_risk_to_plot.groupby(['sens_Var_level'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # y_all_mean = np.round(np.mean(np.array(Avg_risk_score_list)), decimals=4)
    # y_all_std = np.round(np.std(np.array(Avg_risk_score_list)), decimals=4)
    # ax.axhline(y=y_all_mean, xmin=-1, xmax=1, color='brown', linestyle='--', lw=1,
    #            label="Avg_predicted_risk_all")
    # ax.axhline(y=true_outcome_rate, xmin=-1, xmax=1, color='orange', linestyle='--', lw=1,
    #            label="True_outcome_rate_all")
    # for name, group in groups:
    #     print(name)
    #     print(group)
    #     # print(mkr_dict[name[1]])
    #     print(color_dict[name.split("_")[0]])
    #     # marker = mkr_dict[name[1]]
    #     color = color_dict[name.split("_")[0]]
    #     # print(name)
    #     print(" Check point 0")
    #     if name == 'All':
    #         plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none',marker='o')
    #         plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                      fmt='None', capsize=1.5)
    #         print("Check point 1")
    #     elif name.split("_")[0] == 'age':
    #         print("Check point 2")
    #         if (name.split("_")[1] != '0.0') & (
    #                 ctr_legend == 0):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values,
    #                         label='(' + str(name.split("_")[0]) + ')', color=color, linewidth=0,
    #                         ms=8, fillstyle='none',marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none',marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     else:
    #         print("Check point 3")
    #         if ((name.split("_")[1] == str(0.0)) or (name.split("_")[1] == str(1)) or (
    #                 (name.split("_")[1] == str(1.0)))) & (name.split("_")[0] != 'age'):
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values,
    #                         label='(' + str(name.split("_")[0]) + ')', color=color,
    #                         linewidth=0,
    #                         ms=8, fillstyle='none',marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #         else:
    #             plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none',marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #
    # Mean_Pred_risk_list_afterpsm = [
    #     np.round(np.mean(np.array(Average_risk_score_level_wise_list_after_psm[i])), decimals=4)
    #     for i in
    #     Average_risk_score_level_wise_list_after_psm.keys()]
    # STD_Pred_risk_list_afterpsm = [
    #     np.round(np.std(np.array(Average_risk_score_level_wise_list_after_psm[i])), decimals=4)
    #     for i in
    #     Average_risk_score_level_wise_list_after_psm.keys()]
    # Pred_risk_to_plot_psm = pd.DataFrame()
    # Pred_risk_to_plot_psm['sens_Var_level_crude'] = Average_risk_score_level_wise_list_after_psm.keys()
    # Pred_risk_to_plot_psm['Mean_values'] = Mean_Pred_risk_list_afterpsm
    # Pred_risk_to_plot_psm['Std_values'] = STD_Pred_risk_list_afterpsm
    # Pred_risk_to_plot_psm['method'] = " "
    # Pred_risk_to_plot_psm['sens_Var_level'] = " "
    # Pred_risk_to_plot_psm['x_ticks'] = " "
    # Pred_risk_to_plot_psm['x'] = 0
    # dict_with_levels = dict(zip(sens_var_levels, np.arange(len(sens_var_levels))))
    # idx = 0
    # for j in range(len(Average_risk_score_level_wise_list_after_psm.keys())):
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'Sex':
    #         Pred_risk_to_plot_psm.at[idx, 'method'] = "PSM"
    #         Pred_risk_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #             list(Average_risk_score_level_wise_list_after_psm.keys())[j]
    #         Pred_risk_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[
    #             list(Average_risk_score_level_wise_list_after_psm.keys())[j]]
    #         Pred_risk_to_plot_psm.at[idx, 'x'] = dict_with_levels[Pred_risk_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'age':
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[1] == '0psm10':
    #             Pred_risk_to_plot_psm.at[idx, 'method'] = "PSM"
    #             Pred_risk_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #             Pred_risk_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_1-0"
    #             Pred_risk_to_plot_psm.at[idx, 'x'] = dict_with_levels[Pred_risk_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[1] == '0psm12':
    #             Pred_risk_to_plot_psm.at[idx, 'method'] = "PSM1"
    #             Pred_risk_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #             Pred_risk_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_1-2"
    #             Pred_risk_to_plot_psm.at[idx, 'x'] = dict_with_levels[Pred_risk_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'RACE':
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'BW':
    #             Pred_risk_to_plot_psm.at[idx, 'method'] = "PSM"
    #             Pred_risk_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("psm")[0]
    #             Pred_risk_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split("psm")[0]] + "-PSM_B-W"
    #             Pred_risk_to_plot_psm.at[idx, 'x'] = dict_with_levels[Pred_risk_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'OB':
    #             if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[1] == 'nanpsm':
    #                 Pred_risk_to_plot_psm.at[idx, 'method'] = "PSM1"
    #                 Pred_risk_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"
    #                 Pred_risk_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"] + "-PSM_O-B"
    #                 Pred_risk_to_plot_psm.at[idx, 'x'] = dict_with_levels[Pred_risk_to_plot_psm['sens_Var_level'][idx]]
    #             else:
    #                 Pred_risk_to_plot_psm.at[idx, 'method'] = "PSM1"
    #                 Pred_risk_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #                 Pred_risk_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_O-B"
    #                 Pred_risk_to_plot_psm.at[idx, 'x'] = dict_with_levels[Pred_risk_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'OW':
    #             if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[1] == 'nanpsm':
    #                 Pred_risk_to_plot_psm.at[idx, 'method'] = "PSM2"
    #                 Pred_risk_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"
    #                 Pred_risk_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"] + "-PSM_O-W"
    #                 Pred_risk_to_plot_psm.at[idx, 'x'] = dict_with_levels[Pred_risk_to_plot_psm['sens_Var_level'][idx]]
    #             else:
    #                 Pred_risk_to_plot_psm.at[idx, 'method'] = "PSM2"
    #                 Pred_risk_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #                 Pred_risk_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_O-W"
    #                 Pred_risk_to_plot_psm.at[idx, 'x'] = dict_with_levels[Pred_risk_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    # Pred_risk_to_plot_psm['x'] = Pred_risk_to_plot_psm['x'] + 0.15
    #
    # mkr_dict = {'PSM1': '*', 'PSM': '^', 'PSM2': 's'}
    # color_dict = {'Sex': 'r', 'RACE': 'b', 'age': 'g'}
    # groups_psm = Pred_risk_to_plot_psm.groupby(['sens_Var_level', 'method'])
    # ctr_legend = 0
    # for name, group in groups_psm:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name)
    #     if name[0] == 'All':
    #         plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, s=5)
    #         plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                      fmt='None', capsize=1.5)
    #     elif name[0].split("_")[0] == 'age':
    #         if (name[0].split("_")[1] == '1.0') & (
    #                 ctr_legend != 2):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Mid-young)'
    #             else:
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Mid-old)'
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label=label_value, color=color, linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     elif name[0].split("_")[0] == 'RACE':
    #         if (name[0].split("_")[
    #             1] == '1'):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Black-White)'
    #             if name[1] == 'PSM1':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Black-Others)'
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label=label_value, color=color, linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         elif (name[0].split("_")[
    #                   1] == '0'):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM2':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_White-Others)'
    #                 plt.plot(group.x, group.Mean_values, marker=marker,
    #                             label=label_value, color=color, linewidth=0,
    #                             ms=8, fillstyle='none')
    #             else:
    #                 plt.plot(group.x, group.Mean_values, marker=marker,
    #                             color=color, linewidth=0,
    #                             ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     else:
    #         if ((name[0].split("_")[1] == str(0.0)) or (name[0].split("_")[1] == str(1)) or (
    #                 (name[0].split("_")[1] == str(1.0)))) & (name[0].split("_")[0] != 'age'):
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label='(' + str(name[0].split("_")[0]) + ',' + str(name[1]) + ')', color=color,
    #                         linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #
    # ax.axvline(x=sens_level_count[0] - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.axvline(x=(sens_level_count[0] + sens_level_count[
    #     1]) - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.axvline(x=(sens_level_count[0] + sens_level_count[1] + sens_level_count[
    #     2]) - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.legend(ncol=3, prop={'size': 7})
    # ax.set_xticklabels(Pred_risk_to_plot['x_ticks'])
    # plt.xticks(Pred_risk_to_plot['x'], Pred_risk_to_plot['x_ticks'])
    # start, end = ax.get_ylim()
    # ax.tick_params(axis='x', which='major', labelsize=8)
    # ax.tick_params(axis='y', which='major', labelsize=10)
    # ax.yaxis.set_ticks(np.arange(max(0, start - 0.2), min(1.01, end + 0.3), 0.1))
    # plt.xlabel('Sensitive variable levels', fontsize=7)
    # plt.ylabel('Average predicted risk', fontsize=7)
    # plt.title(" Average predicted risk for " + str(task))
    # plt.savefig(fig_saving_dir+'/Pred_risk_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #             bbox_inches='tight')
    # plt.savefig(fig_saving_dir+'/Pred_risk_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #             bbox_inches='tight')
    # plt.close()
    #
    # #######################################################################################
    #
    # Mean_Pred_outcome_rate_list = []
    # STD_Pred_outcome_rate_list = []
    # ctr_sv = 0
    # for sv in sens_var:
    #     ctr_l = 0
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             Mean_Pred_outcome_rate_list.append(
    #                 np.round(np.mean(Pred_outcome_rate_level_wise[:, ctr_l, ctr_sv]), decimals=4))
    #             # Mean_Pred_outcome_rate_list_with_after_psm_values.append(
    #             #     np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #             STD_Pred_outcome_rate_list.append(
    #                 np.round(np.std(Pred_outcome_rate_level_wise[:, ctr_l, ctr_sv]), decimals=4))
    #             # STD_Pred_outcome_rate_list_with_after_psm_values.append(
    #             #     np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #             ctr_l = ctr_l + 1
    #     ctr_sv = ctr_sv + 1
    # Pred_outcome_rate_to_plot = pd.DataFrame()
    # Pred_outcome_rate_to_plot['Mean_values'] = Mean_Pred_outcome_rate_list
    # Pred_outcome_rate_to_plot['Std_values'] = STD_Pred_outcome_rate_list
    # Pred_outcome_rate_to_plot['method'] = " "
    # Pred_outcome_rate_to_plot['sens_Var_level'] = " "
    # Pred_outcome_rate_to_plot['x_ticks'] = " "
    # idx = 0
    # for j in range(len(sens_var_levels)):
    #     Pred_outcome_rate_to_plot.at[idx, 'method'] = "Basic"
    #     Pred_outcome_rate_to_plot.at[idx, 'sens_Var_level'] = sens_var_levels[j]
    #     Pred_outcome_rate_to_plot.at[idx, 'x_ticks'] = sens_var_level_dict[sens_var_levels[j]]
    #     idx = idx + 1
    # # Pred_outcome_rate_to_plot.drop(Pred_outcome_rate_to_plot[Pred_outcome_rate_to_plot['Mean_values'] == 0].index,
    # #                                inplace=True)
    # Pred_outcome_rate_to_plot['x'] = np.arange(len(Pred_outcome_rate_to_plot))
    # mkr_dict = {'Basic': 'o', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'RACE': 'b', 'age': 'g', 'All': 'orange'}
    # groups = Pred_outcome_rate_to_plot.groupby(['sens_Var_level'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # y_all_mean = np.round(np.mean(Pred_outcome_rate), decimals=4)
    # y_all_std = np.round(np.std(Pred_outcome_rate), decimals=4)
    # ax.axhline(y=y_all_mean, xmin=-1, xmax=1, color='brown', linestyle='--', lw=1,
    #            label="Predicted_outcome_rate_all")
    # ax.axhline(y=true_outcome_rate, xmin=-1, xmax=1, color='orange', linestyle='--', lw=1,
    #            label="True_outcome_rate_all")
    # plt.fill_between(np.arange(len(Pred_outcome_rate_to_plot)), y_all_mean - y_all_std, y_all_mean + y_all_std,
    #                  color='lightcyan')
    # for name, group in groups:
    #     print(name)
    #     print(group)
    #     # print(mkr_dict[name[1]])
    #     print(color_dict[name.split("_")[0]])
    #     # marker = mkr_dict[name[1]]
    #     color = color_dict[name.split("_")[0]]
    #     # print(name)
    #     print(" Check point 0")
    #     if name == 'All':
    #         plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none' ,marker='o')
    #         plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                      fmt='None', capsize=1.5)
    #         print("Check point 1")
    #     elif name.split("_")[0] == 'age':
    #         print("Check point 2")
    #         if (name.split("_")[1] != '0.0') & (
    #                 ctr_legend == 0):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values,
    #                         label='(' + str(name.split("_")[0]) + ')', color=color, linewidth=0,
    #                         ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     else:
    #         print("Check point 3")
    #         if ((name.split("_")[1] == str(0.0)) or (name.split("_")[1] == str(1)) or (
    #                 (name.split("_")[1] == str(1.0)))) & (name.split("_")[0] != 'age'):
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values,
    #                         label='(' + str(name.split("_")[0]) + ')', color=color,
    #                         linewidth=0,
    #                         ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #         else:
    #             plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #
    # # Mean_Pred_outcome_rate_list_with_after_psm_values = []
    # # STD_Pred_outcome_rate_list_with_after_psm_values = []
    # # ctr_sv = 0
    # # for sv in sens_var:
    # #     ctr_l = 0
    # #     for name in sens_var_levels:
    # #         if name.split("_")[0] == sv:
    # #             if sv == 'age':  # being treated differently because there are two case of psm match in age
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    # #                       np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 Mean_Pred_outcome_rate_list_with_after_psm_values.append(
    # #                     np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 STD_Pred_outcome_rate_list_with_after_psm_values.append(
    # #                     np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4),
    # #                       np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 Mean_Pred_outcome_rate_list_with_after_psm_values.append(
    # #                     np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 STD_Pred_outcome_rate_list_with_after_psm_values.append(
    # #                     np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #             elif sv == 'Race':  # being treated differently because there are two case of psm match in age
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    # #                       np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 Mean_Pred_outcome_rate_list_with_after_psm_values.append(
    # #                     np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 STD_Pred_outcome_rate_list_with_after_psm_values.append(
    # #                     np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4),
    # #                       np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 Mean_Pred_outcome_rate_list_with_after_psm_values.append(
    # #                     np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 STD_Pred_outcome_rate_list_with_after_psm_values.append(
    # #                     np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4),
    # #                       np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4))
    # #                 Mean_Pred_outcome_rate_list_with_after_psm_values.append(
    # #                     np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4))
    # #                 STD_Pred_outcome_rate_list_with_after_psm_values.append(
    # #                     np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4))
    # #             else:
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    # #                       np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 Mean_Pred_outcome_rate_list_with_after_psm_values.append(
    # #                     np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 STD_Pred_outcome_rate_list_with_after_psm_values.append(
    # #                     np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #             ctr_l = ctr_l + 1
    # #     if sv == 'age':
    # #         ctr_sv = ctr_sv + 2
    # #     if sv == 'Race':
    # #         ctr_sv = ctr_sv + 3
    # #     if sv == 'Sex':
    # #         ctr_sv = ctr_sv + 1
    #
    # Pred_outcome_rate_average_across_bts_level_wise_psm = np.mean(Pred_outcome_rate_level_wise_after_psm, axis=0)
    # Pred_outcome_rate_std_across_bts_level_wise_psm = np.std(Pred_outcome_rate_level_wise_after_psm, axis=0)
    #
    # Mean_Pred_outcome_rate_list_with_after_psm_values = [np.round(Pred_outcome_rate_average_across_bts_level_wise_psm[i,j], decimals=4) for j in range(6) for i in range(3)]
    # STD_Pred_outcome_rate_list_with_after_psm_values = [np.round(Pred_outcome_rate_std_across_bts_level_wise_psm[i,j], decimals=4) for j in range(6) for i in range(3)]
    #
    #
    # Mean_Pred_outcome_rate_list_with_after_psm_values = list(
    #     filter(lambda num: num != 0, Mean_Pred_outcome_rate_list_with_after_psm_values))
    # STD_Pred_outcome_rate_list_with_after_psm_values = list(
    #     filter(lambda num: num != 0, STD_Pred_outcome_rate_list_with_after_psm_values))
    # Pred_outcome_rate_to_plot_psm = pd.DataFrame()
    # Pred_outcome_rate_to_plot_psm['sens_Var_level_crude'] = Average_risk_score_level_wise_list_after_psm.keys()
    # Pred_outcome_rate_to_plot_psm['Mean_values'] = Mean_Pred_outcome_rate_list_with_after_psm_values
    # Pred_outcome_rate_to_plot_psm['Std_values'] = STD_Pred_outcome_rate_list_with_after_psm_values
    # Pred_outcome_rate_to_plot_psm['method'] = " "
    # Pred_outcome_rate_to_plot_psm['sens_Var_level'] = " "
    # Pred_outcome_rate_to_plot_psm['x_ticks'] = " "
    # Pred_outcome_rate_to_plot_psm['x'] = 0
    # dict_with_levels = dict(zip(sens_var_levels, np.arange(len(sens_var_levels))))
    # idx = 0
    # for j in range(len(Average_risk_score_level_wise_list_after_psm.keys())):
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'Sex':
    #         Pred_outcome_rate_to_plot_psm.at[idx, 'method'] = "PSM"
    #         Pred_outcome_rate_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #             list(Average_risk_score_level_wise_list_after_psm.keys())[j]
    #         Pred_outcome_rate_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[
    #             list(Average_risk_score_level_wise_list_after_psm.keys())[j]]
    #         Pred_outcome_rate_to_plot_psm.at[idx, 'x'] = dict_with_levels[
    #             Pred_outcome_rate_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'age':
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[1] == '0psm10':
    #             Pred_outcome_rate_to_plot_psm.at[idx, 'method'] = "PSM"
    #             Pred_outcome_rate_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #             Pred_outcome_rate_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_1-0"
    #             Pred_outcome_rate_to_plot_psm.at[idx, 'x'] = dict_with_levels[
    #                 Pred_outcome_rate_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[1] == '0psm12':
    #             Pred_outcome_rate_to_plot_psm.at[idx, 'method'] = "PSM1"
    #             Pred_outcome_rate_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #             Pred_outcome_rate_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_1-2"
    #             Pred_outcome_rate_to_plot_psm.at[idx, 'x'] = dict_with_levels[
    #                 Pred_outcome_rate_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'RACE':
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'BW':
    #             Pred_outcome_rate_to_plot_psm.at[idx, 'method'] = "PSM"
    #             Pred_outcome_rate_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("psm")[0]
    #             Pred_outcome_rate_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split("psm")[0]] + "-PSM_B-W"
    #             Pred_outcome_rate_to_plot_psm.at[idx, 'x'] = dict_with_levels[
    #                 Pred_outcome_rate_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'OB':
    #             if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[1] == 'nanpsm':
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'method'] = "PSM1"
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"] + "-PSM_O-B"
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'x'] = dict_with_levels[
    #                     Pred_outcome_rate_to_plot_psm['sens_Var_level'][idx]]
    #             else:
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'method'] = "PSM1"
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_O-B"
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'x'] = dict_with_levels[
    #                     Pred_outcome_rate_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'OW':
    #             if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[1] == 'nanpsm':
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'method'] = "PSM2"
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"] + "-PSM_O-W"
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'x'] = dict_with_levels[
    #                     Pred_outcome_rate_to_plot_psm['sens_Var_level'][idx]]
    #             else:
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'method'] = "PSM2"
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_O-W"
    #                 Pred_outcome_rate_to_plot_psm.at[idx, 'x'] = dict_with_levels[
    #                     Pred_outcome_rate_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    # Pred_outcome_rate_to_plot_psm['x'] = Pred_outcome_rate_to_plot_psm['x'] + 0.15
    #
    # mkr_dict = {'PSM1': '*', 'PSM': '^', 'PSM2': 's'}
    # color_dict = {'Sex': 'r', 'RACE': 'b', 'age': 'g'}
    # groups_psm = Pred_outcome_rate_to_plot_psm.groupby(['sens_Var_level', 'method'])
    # ctr_legend = 0
    # for name, group in groups_psm:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name)
    #     if name[0] == 'All':
    #         plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, s=5)
    #         plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                      fmt='None', capsize=1.5)
    #     elif name[0].split("_")[0] == 'age':
    #         if (name[0].split("_")[1] == '1.0') & (
    #                 ctr_legend != 2):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Mid-young)'
    #             else:
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Mid-old)'
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label=label_value, color=color, linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     elif name[0].split("_")[0] == 'RACE':
    #         if (name[0].split("_")[
    #             1] == '1'):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Black-White)'
    #             if name[1] == 'PSM1':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Black-Others)'
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label=label_value, color=color, linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         elif (name[0].split("_")[
    #                   1] == '0'):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM2':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_White-Others)'
    #                 plt.plot(group.x, group.Mean_values, marker=marker,
    #                             label=label_value, color=color, linewidth=0,
    #                             ms=8, fillstyle='none')
    #             else:
    #                 plt.plot(group.x, group.Mean_values, marker=marker,
    #                             color=color, linewidth=0,
    #                             ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     else:
    #         if ((name[0].split("_")[1] == str(0.0)) or (name[0].split("_")[1] == str(1)) or (
    #                 (name[0].split("_")[1] == str(1.0)))) & (name[0].split("_")[0] != 'age'):
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label='(' + str(name[0].split("_")[0]) + ',' + str(name[1]) + ')', color=color,
    #                         linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    # ax.axvline(x=sens_level_count[0] - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.axvline(x=(sens_level_count[0] + sens_level_count[
    #     1]) - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.axvline(x=(sens_level_count[0] + sens_level_count[1] + sens_level_count[
    #     2]) - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.legend(ncol=3, prop={'size': 7})
    # ax.set_xticklabels(Pred_outcome_rate_to_plot['x_ticks'])
    # plt.xticks(Pred_outcome_rate_to_plot['x'], Pred_outcome_rate_to_plot['x_ticks'])
    # start, end = ax.get_ylim()
    # ax.tick_params(axis='x', which='major', labelsize=10)
    # ax.tick_params(axis='y', which='major', labelsize=10)
    # ax.yaxis.set_ticks(np.arange(max(0, start - 0.2), min(1.01, end + 0.3), 0.1))
    # plt.xlabel('Sensitive variable levels', fontsize=7)
    # plt.ylabel('Pred outcome rate values', fontsize=7)
    # plt.title(" Predicted outcome rate values @ " + str(np.round(threshold[0], decimals=3)))
    # plt.savefig(fig_saving_dir+'/Pred_outcome_rate_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #             bbox_inches='tight')
    # plt.savefig(fig_saving_dir+'/Pred_outcome_rate_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #             bbox_inches='tight')
    # plt.close()
    # ###### alternative plotting tech
    # """
    # Pred_outcome_rate_to_plot = Pred_outcome_rate_to_plot.append(
    #     {'Mean_values': np.round(np.mean(Pred_outcome_rate), decimals=4),
    #      'Std_values': np.round(np.std(Pred_outcome_rate), decimals=4), 'method': 'Basic', 'sens_Var_level': 'All',
    #      'x_ticks': 'All', 'x': len(Pred_outcome_rate_to_plot) }, ignore_index=True)
    #
    # mkr_dict = {'Basic': 'o', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'Race': 'b', 'age': 'g', 'All': 'orange'}
    # groups = Pred_outcome_rate_to_plot.groupby(['sens_Var_level'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # for name, group in groups:
    #     # marker = mkr_dict[name[1]]
    #     color = color_dict[name.split("_")[0]]
    #     # print(name, color)
    #     plt.scatter(group.Mean_values, group.x, color=color, linewidths=0,
    #                 s=70)
    #     plt.errorbar(group.Mean_values, group.x, xerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                  fmt='None', capsize=1.5)
    #
    # mkr_dict = {'PSM1': '*', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'Race': 'b', 'age': 'g'}
    # groups_psm = Pred_outcome_rate_to_plot_psm.groupby(['sens_Var_level', 'method'])
    # # fig, ax = plt.subplots()
    # # ax.margins(0.05)
    # ctr_legend = 0
    # for name, group in groups_psm:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name, color)
    #     plt.scatter(group.Mean_values, group.x, marker=marker, color=color, linewidths=0,
    #                 s=70)
    #     plt.errorbar(group.Mean_values, group.x, xerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                  fmt='None', capsize=1.5)
    #
    # ax.axhline(y=sens_level_count[0] - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    # ax.axhline(y=(sens_level_count[0] + sens_level_count[1]) - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    #
    # ax.legend(handles=patches, ncol=4, prop={'size': 7})
    # ax.set_yticklabels(Pred_outcome_rate_to_plot['x_ticks'], ha='right')
    # plt.yticks(Pred_outcome_rate_to_plot['x'], Pred_outcome_rate_to_plot['x_ticks'])
    # start, end = ax.get_xlim()
    # ax.tick_params(axis='x', which='major', labelsize=10)
    # ax.tick_params(axis='y', which='major', labelsize=10)
    # plt.title(" Predicted outcome rate values @ " + str(np.round(threshold[0], decimals=3)))
    # plt.savefig(fig_saving_dir+'/Horizontal_Pred_outcome_rate_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #             bbox_inches='tight')
    # plt.savefig(fig_saving_dir+'/Horizontal_Pred_outcome_rate_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #             bbox_inches='tight')
    # plt.close()
    #
    # Mean_Pred_outcome_rate_list_with_after_psm_values_Horizontal = []
    # STD_Pred_outcome_rate_list_with_after_psm_values_Horizontal = []
    # ctr_sv = 0
    # for sv in sens_var:
    #     ctr_l = 0
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             Mean_Pred_outcome_rate_list_with_after_psm_values_Horizontal.append(
    #                 np.round(np.mean(Pred_outcome_rate_level_wise[:, ctr_l, ctr_sv]), decimals=4))
    #             STD_Pred_outcome_rate_list_with_after_psm_values_Horizontal.append(
    #                 np.round(np.std(Pred_outcome_rate_level_wise[:, ctr_l, ctr_sv]), decimals=4))
    #             ctr_l = ctr_l + 1
    #     ctr_l = 0
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             if sv == 'age':  # being treated differently because there are two case of psm match in age
    #                 print(" Counter values ", ctr_sv, ctr_l)
    #                 print(sv, name,
    #                       np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    #                       np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 Mean_Pred_outcome_rate_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 STD_Pred_outcome_rate_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 print(" Counter values ", ctr_sv, ctr_l)
    #                 print(sv, name,
    #                       np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4),
    #                       np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    #                 Mean_Pred_outcome_rate_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    #                 STD_Pred_outcome_rate_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    #             else:
    #                 Mean_Pred_outcome_rate_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.mean(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 STD_Pred_outcome_rate_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.std(Pred_outcome_rate_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #             ctr_l = ctr_l + 1
    #     ctr_sv = ctr_sv + 1
    #
    # Pred_outcome_rate_to_plot_Horizontal = pd.DataFrame()
    # Pred_outcome_rate_to_plot_Horizontal['Mean_values'] = Mean_Pred_outcome_rate_list_with_after_psm_values_Horizontal
    # Pred_outcome_rate_to_plot_Horizontal['Std_values'] = STD_Pred_outcome_rate_list_with_after_psm_values_Horizontal
    # Pred_outcome_rate_to_plot_Horizontal['method'] = " "
    # Pred_outcome_rate_to_plot_Horizontal['sens_Var_level'] = " "
    # Pred_outcome_rate_to_plot_Horizontal['x_ticks'] = " "
    # idx = 0
    # idx_for_level = 0
    # for sv in sens_var:
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             Pred_outcome_rate_to_plot_Horizontal.at[idx, 'method'] = "Basic"
    #             Pred_outcome_rate_to_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #             Pred_outcome_rate_to_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[name]
    #             idx = idx + 1
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             if sv == 'age':  # being treated differently because there are two case of psm match in age
    #                 Pred_outcome_rate_to_plot_Horizontal.at[idx, 'method'] = "PSM"
    #                 Pred_outcome_rate_to_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #                 Pred_outcome_rate_to_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[
    #                                                                               name] + "PSM_Mid-young"
    #                 idx = idx + 1
    #                 Pred_outcome_rate_to_plot_Horizontal.at[idx, 'method'] = "PSM1"
    #                 Pred_outcome_rate_to_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #                 Pred_outcome_rate_to_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[
    #                                                                               name] + "PSM_Mid-old"
    #                 idx = idx + 1
    #             else:
    #                 Pred_outcome_rate_to_plot_Horizontal.at[idx, 'method'] = "PSM"
    #                 Pred_outcome_rate_to_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #                 Pred_outcome_rate_to_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[
    #                                                                               name] + "-PSM"
    #                 idx = idx + 1
    #
    # Pred_outcome_rate_to_plot_Horizontal.drop(
    #     Pred_outcome_rate_to_plot_Horizontal[Pred_outcome_rate_to_plot_Horizontal['Mean_values'] == 0].index,
    #     inplace=True)
    # Pred_outcome_rate_to_plot_Horizontal = Pred_outcome_rate_to_plot_Horizontal.append(
    #     {'Mean_values': np.round(np.mean(Pred_outcome_rate), decimals=4),
    #      'Std_values': np.round(np.std(Pred_outcome_rate), decimals=4), 'method': 'Basic', 'sens_Var_level': 'All',
    #      'x_ticks': 'All', }, ignore_index=True)
    # Pred_outcome_rate_to_plot_Horizontal['x'] = np.arange(len(Pred_outcome_rate_to_plot_Horizontal))
    #
    # mkr_dict = {'Basic': 'o', 'PSM': '^', 'PSM1': '*'}
    # color_dict = {'Sex': 'r', 'Race': 'b', 'age': 'g', 'All': 'orange'}
    # groups = Pred_outcome_rate_to_plot_Horizontal.groupby(['sens_Var_level', 'method'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # for name, group in groups:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name, color)
    #     plt.scatter(group.Mean_values, group.x, marker=marker, color=color, linewidths=0,
    #                 s=70)
    #     plt.errorbar(group.Mean_values, group.x, xerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                  fmt='None', capsize=1.5)
    #
    # ax.axhline(y=sens_level_count[0] * 2 - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    # ax.axhline(y=(sens_level_count[0] * 2 + sens_level_count[1] * 2) - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    #
    # ax.legend(handles=patches, ncol=4, prop={'size': 7})
    # ax.set_yticklabels(Pred_outcome_rate_to_plot_Horizontal['x_ticks'], ha='right')
    # plt.yticks(Pred_outcome_rate_to_plot_Horizontal['x'], Pred_outcome_rate_to_plot_Horizontal['x_ticks'])
    # start, end = ax.get_xlim()
    # ax.tick_params(axis='x', which='major', labelsize=10)
    # ax.tick_params(axis='y', which='major', labelsize=6)
    # plt.title(" Predicted outcome rate values @ " + str(np.round(threshold[0], decimals=3)))
    # plt.savefig(
    #     fig_saving_dir+'/Non-stacked_Horizontal_Pred_outcome_rate_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #     bbox_inches='tight')
    # plt.savefig(
    #     fig_saving_dir+'/Non-stacked_Horizontal_Pred_outcome_rate_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #     bbox_inches='tight')
    # plt.close()
    # """
    # # Accuracy plots
    #
    # Mean_accuracy_list = []
    # STD_accuracy_list = []
    # ctr_sv = 0
    # for sv in sens_var:
    #     ctr_l = 0
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             Mean_accuracy_list.append(
    #                 np.round(np.mean(ACC_level_wise[:, ctr_l, ctr_sv]), decimals=4))
    #             # Mean_accuracy_list_with_after_psm_values.append(
    #             #     np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #             STD_accuracy_list.append(
    #                 np.round(np.std(ACC_level_wise[:, ctr_l, ctr_sv]), decimals=4))
    #             # STD_accuracy_list_with_after_psm_values.append(
    #             #     np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #             ctr_l = ctr_l + 1
    #     ctr_sv = ctr_sv + 1
    # Accuracy_to_plot = pd.DataFrame()
    # Accuracy_to_plot['Mean_values'] = Mean_accuracy_list
    # Accuracy_to_plot['Std_values'] = STD_accuracy_list
    # Accuracy_to_plot['method'] = " "
    # Accuracy_to_plot['sens_Var_level'] = " "
    # Accuracy_to_plot['x_ticks'] = " "
    # idx = 0
    # for j in range(len(sens_var_levels)):
    #     Accuracy_to_plot.at[idx, 'method'] = "Basic"
    #     Accuracy_to_plot.at[idx, 'sens_Var_level'] = sens_var_levels[j]
    #     Accuracy_to_plot.at[idx, 'x_ticks'] = sens_var_level_dict[sens_var_levels[j]]
    #     idx = idx + 1
    # # Accuracy_to_plot.drop(Accuracy_to_plot[Accuracy_to_plot['Mean_values'] == 0].index,
    # #                                inplace=True)
    # Accuracy_to_plot['x'] = np.arange(len(Accuracy_to_plot))
    # mkr_dict = {'Basic': 'o', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'RACE': 'b', 'age': 'g', 'All': 'orange'}
    # groups = Accuracy_to_plot.groupby(['sens_Var_level'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # y_all_mean = np.round(np.mean(ACC), decimals=4)
    # y_all_std = np.round(np.std(ACC), decimals=4)
    # ax.axhline(y=y_all_mean, xmin=-1, xmax=1, color='brown', linestyle='--', lw=1,
    #            label="All")
    # # ax.axhline(y=true_outcome_rate, xmin=-1, xmax=1, color='orange', linestyle='--', lw=1,
    # #            label="True_outcome_rate_all")
    # plt.fill_between(np.arange(len(Accuracy_to_plot)), y_all_mean - y_all_std, y_all_mean + y_all_std,
    #                  color='lightcyan')
    # for name, group in groups:
    #     print(name)
    #     print(group)
    #     # print(mkr_dict[name[1]])
    #     print(color_dict[name.split("_")[0]])
    #     # marker = mkr_dict[name[1]]
    #     color = color_dict[name.split("_")[0]]
    #     # print(name)
    #     print(" Check point 0")
    #     if name == 'All':
    #         plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none' ,marker='o')
    #         plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                      fmt='None', capsize=1.5)
    #         print("Check point 1")
    #     elif name.split("_")[0] == 'age':
    #         print("Check point 2")
    #         if (name.split("_")[1] != '0.0') & (
    #                 ctr_legend == 0):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values,
    #                         label='(' + str(name.split("_")[0]) + ')', color=color, linewidth=0,
    #                         ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     else:
    #         print("Check point 3")
    #         if ((name.split("_")[1] == str(0.0)) or (name.split("_")[1] == str(1)) or (
    #                 (name.split("_")[1] == str(1.0)))) & (name.split("_")[0] != 'age'):
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values,
    #                         label='(' + str(name.split("_")[0]) + ')', color=color,
    #                         linewidth=0,
    #                         ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #         else:
    #             plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #
    # # Mean_ACC_list_with_after_psm_values = []
    # # STD_ACC_list_with_after_psm_values = []
    # # ctr_sv = 0
    # # for sv in sens_var:
    # #     ctr_l = 0
    # #     for name in sens_var_levels:
    # #         if name.split("_")[0] == sv:
    # #             if sv == 'age':  # being treated differently because there are two case of psm match in age
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    # #                       np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 Mean_ACC_list_with_after_psm_values.append(
    # #                     np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 STD_ACC_list_with_after_psm_values.append(
    # #                     np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4),
    # #                       np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 Mean_ACC_list_with_after_psm_values.append(
    # #                     np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 STD_ACC_list_with_after_psm_values.append(
    # #                     np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #             elif sv == 'Race':  # being treated differently because there are two case of psm match in age
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    # #                       np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 Mean_ACC_list_with_after_psm_values.append(
    # #                     np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 STD_ACC_list_with_after_psm_values.append(
    # #                     np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4),
    # #                       np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 Mean_ACC_list_with_after_psm_values.append(
    # #                     np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 STD_ACC_list_with_after_psm_values.append(
    # #                     np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4),
    # #                       np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4))
    # #                 Mean_ACC_list_with_after_psm_values.append(
    # #                     np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4))
    # #                 STD_ACC_list_with_after_psm_values.append(
    # #                     np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4))
    # #             else:
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    # #                       np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 Mean_ACC_list_with_after_psm_values.append(
    # #                     np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 STD_ACC_list_with_after_psm_values.append(
    # #                     np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #             ctr_l = ctr_l + 1
    # #     if sv == 'age':
    # #         ctr_sv = ctr_sv + 2
    # #     if sv == 'Race':
    # #         ctr_sv = ctr_sv + 3
    # #     if sv == 'Sex':
    # #         ctr_sv = ctr_sv + 1
    #
    #
    # ACC_average_across_bts_level_wise_psm = np.mean(ACC_level_wise_after_psm, axis=0)
    # ACC_std_across_bts_level_wise_psm = np.std(ACC_level_wise_after_psm, axis=0)
    #
    # Mean_ACC_list_with_after_psm_values = [np.round(ACC_average_across_bts_level_wise_psm[i,j], decimals=4) for j in range(6) for i in range(3)]
    # STD_ACC_list_with_after_psm_values = [np.round(ACC_std_across_bts_level_wise_psm[i,j], decimals=4) for j in range(6) for i in range(3)]
    #
    #
    # Mean_ACC_list_with_after_psm_values = list(filter(lambda num: num != 0, Mean_ACC_list_with_after_psm_values))
    # STD_ACC_list_with_after_psm_values = list(filter(lambda num: num != 0, STD_ACC_list_with_after_psm_values))
    # Accuracy_to_plot_psm = pd.DataFrame()
    # Accuracy_to_plot_psm['sens_Var_level_crude'] = Average_risk_score_level_wise_list_after_psm.keys()
    # Accuracy_to_plot_psm['Mean_values'] = Mean_ACC_list_with_after_psm_values
    # Accuracy_to_plot_psm['Std_values'] = STD_ACC_list_with_after_psm_values
    # Accuracy_to_plot_psm['method'] = " "
    # Accuracy_to_plot_psm['sens_Var_level'] = " "
    # Accuracy_to_plot_psm['x_ticks'] = " "
    # Accuracy_to_plot_psm['x'] = 0
    # dict_with_levels = dict(zip(sens_var_levels, np.arange(len(sens_var_levels))))
    # idx = 0
    # for j in range(len(Average_risk_score_level_wise_list_after_psm.keys())):
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'Sex':
    #         Accuracy_to_plot_psm.at[idx, 'method'] = "PSM"
    #         Accuracy_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #             list(Average_risk_score_level_wise_list_after_psm.keys())[j]
    #         Accuracy_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[
    #             list(Average_risk_score_level_wise_list_after_psm.keys())[j]]
    #         Accuracy_to_plot_psm.at[idx, 'x'] = dict_with_levels[Accuracy_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'age':
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[1] == '0psm10':
    #             Accuracy_to_plot_psm.at[idx, 'method'] = "PSM"
    #             Accuracy_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("psm")[0]
    #             Accuracy_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split("psm")[0]] + "-PSM_1-0"
    #             Accuracy_to_plot_psm.at[idx, 'x'] = dict_with_levels[Accuracy_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[1] == '0psm12':
    #             Accuracy_to_plot_psm.at[idx, 'method'] = "PSM1"
    #             Accuracy_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #             Accuracy_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_1-2"
    #             Accuracy_to_plot_psm.at[idx, 'x'] = dict_with_levels[Accuracy_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'RACE':
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'BW':
    #             Accuracy_to_plot_psm.at[idx, 'method'] = "PSM"
    #             Accuracy_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("psm")[0]
    #             Accuracy_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split("psm")[0]] + "-PSM_B-W"
    #             Accuracy_to_plot_psm.at[idx, 'x'] = dict_with_levels[Accuracy_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'OB':
    #             if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[1] == 'nanpsm':
    #                 Accuracy_to_plot_psm.at[idx, 'method'] = "PSM1"
    #                 Accuracy_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"
    #                 Accuracy_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"] + "-PSM_O-B"
    #                 Accuracy_to_plot_psm.at[idx, 'x'] = dict_with_levels[Accuracy_to_plot_psm['sens_Var_level'][idx]]
    #             else:
    #                 Accuracy_to_plot_psm.at[idx, 'method'] = "PSM1"
    #                 Accuracy_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #                 Accuracy_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_O-B"
    #                 Accuracy_to_plot_psm.at[idx, 'x'] = dict_with_levels[Accuracy_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'OW':
    #             if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[1] == 'nanpsm':
    #                 Accuracy_to_plot_psm.at[idx, 'method'] = "PSM2"
    #                 Accuracy_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"
    #                 Accuracy_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"] + "-PSM_O-W"
    #                 Accuracy_to_plot_psm.at[idx, 'x'] = dict_with_levels[Accuracy_to_plot_psm['sens_Var_level'][idx]]
    #             else:
    #                 Accuracy_to_plot_psm.at[idx, 'method'] = "PSM2"
    #                 Accuracy_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #                 Accuracy_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_O-W"
    #                 Accuracy_to_plot_psm.at[idx, 'x'] = dict_with_levels[Accuracy_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    # Accuracy_to_plot_psm['x'] = Accuracy_to_plot_psm['x'] + 0.15
    #
    # mkr_dict = {'PSM1': '*', 'PSM': '^', 'PSM2': 's'}
    # color_dict = {'Sex': 'r', 'RACE': 'b', 'age': 'g'}
    # groups_psm = Accuracy_to_plot_psm.groupby(['sens_Var_level', 'method'])
    # ctr_legend = 0
    # for name, group in groups_psm:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name)
    #     if name[0] == 'All':
    #         plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, s=5)
    #         plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                      fmt='None', capsize=1.5)
    #     elif name[0].split("_")[0] == 'age':
    #         if (name[0].split("_")[1] == '1.0') & (
    #                 ctr_legend != 2):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Mid-young)'
    #             else:
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Mid-old)'
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label=label_value, color=color, linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     elif name[0].split("_")[0] == 'RACE':
    #         if (name[0].split("_")[
    #             1] == '1'):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Black-White)'
    #             if name[1] == 'PSM1':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Black-Others)'
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label=label_value, color=color, linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         elif (name[0].split("_")[
    #                   1] == '0'):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM2':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_White-Others)'
    #                 plt.plot(group.x, group.Mean_values, marker=marker,
    #                             label=label_value, color=color, linewidth=0,
    #                             ms=8, fillstyle='none')
    #             else:
    #                 plt.plot(group.x, group.Mean_values, marker=marker,
    #                             color=color, linewidth=0,
    #                             ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     else:
    #         if ((name[0].split("_")[1] == str(0.0)) or (name[0].split("_")[1] == str(1)) or (
    #                 (name[0].split("_")[1] == str(1.0)))) & (name[0].split("_")[0] != 'age'):
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label='(' + str(name[0].split("_")[0]) + ',' + str(name[1]) + ')', color=color,
    #                         linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    # ax.axvline(x=sens_level_count[0] - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.axvline(x=(sens_level_count[0] + sens_level_count[
    #     1]) - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.axvline(x=(sens_level_count[0] + sens_level_count[1] + sens_level_count[
    #     2]) - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.legend(ncol=3, prop={'size': 7})
    # ax.set_xticklabels(Accuracy_to_plot['x_ticks'])
    # plt.xticks(Accuracy_to_plot['x'], Accuracy_to_plot['x_ticks'])
    # start, end = ax.get_ylim()
    # ax.tick_params(axis='x', which='major', labelsize=8)
    # ax.tick_params(axis='y', which='major', labelsize=10)
    # ax.yaxis.set_ticks(np.arange(max(0, start - 0.2), min(1.01, end + 0.3), 0.1))
    # plt.xlabel('Sensitive variable levels', fontsize=7)
    # plt.ylabel('Accuracy values', fontsize=7)
    # plt.title(" Accuracy values @ " + str(np.round(threshold[0], decimals=3)))
    # plt.savefig(fig_saving_dir+'/Accuracy_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #             bbox_inches='tight')
    # plt.savefig(fig_saving_dir+'/Accuracy_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #             bbox_inches='tight')
    # plt.close()
    # ###### alternative plotting tech
    # """
    # Accuracy_to_plot = Accuracy_to_plot.append(
    #     {'Mean_values': np.round(np.mean(ACC), decimals=4),
    #      'Std_values': np.round(np.std(ACC), decimals=4), 'method': 'Basic', 'sens_Var_level': 'All',
    #      'x_ticks': 'All', 'x': len(Accuracy_to_plot)}, ignore_index=True)
    # mkr_dict = {'Basic': 'o', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'Race': 'b', 'age': 'g', 'All': 'orange'}
    # groups = Accuracy_to_plot.groupby(['sens_Var_level'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # for name, group in groups:
    #     # marker = mkr_dict[name[1]]
    #     color = color_dict[name.split("_")[0]]
    #     # print(name, color)
    #     plt.scatter(group.Mean_values, group.x, color=color, linewidths=0,
    #                 ms=8, fillstyle='none')
    #     plt.errorbar(group.Mean_values, group.x, xerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                  fmt='None', capsize=1.5)
    # mkr_dict = {'PSM1': '*', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'Race': 'b', 'age': 'g'}
    # groups_psm = Accuracy_to_plot_psm.groupby(['sens_Var_level', 'method'])
    # # fig, ax = plt.subplots()
    # # ax.margins(0.05)
    # ctr_legend = 0
    # for name, group in groups_psm:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name, color)
    #     plt.scatter(group.Mean_values, group.x, marker=marker, color=color, linewidths=0,
    #                 s=70)
    #     plt.errorbar(group.Mean_values, group.x, xerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                  fmt='None', capsize=1.5)
    # ax.axhline(y=sens_level_count[0] - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    # ax.axhline(y=(sens_level_count[0] + sens_level_count[1]) - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    # ax.legend(handles=patches, ncol=4, prop={'size': 7})
    # ax.set_yticklabels(Accuracy_to_plot['x_ticks'], ha='right')
    # plt.yticks(Accuracy_to_plot['x'], Accuracy_to_plot['x_ticks'])
    # start, end = ax.get_xlim()
    # ax.tick_params(axis='x', which='major', labelsize=10)
    # ax.tick_params(axis='y', which='major', labelsize=10)
    # plt.title(" Accuracy values @ " + str(np.round(threshold[0], decimals=3)))
    # plt.savefig(fig_saving_dir+'/Horizontal_Accuracy_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #             bbox_inches='tight')
    # plt.savefig(fig_saving_dir+'/Horizontal_Accuracy_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #             bbox_inches='tight')
    # plt.close()
    #
    #
    # Mean_accuracy_list_with_after_psm_values_Horizontal = []
    # STD_accuracy_list_with_after_psm_values_Horizontal = []
    # ctr_sv = 0
    # for sv in sens_var:
    #     ctr_l = 0
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             Mean_accuracy_list_with_after_psm_values_Horizontal.append(
    #                 np.round(np.mean(ACC_level_wise[:, ctr_l, ctr_sv]), decimals=4))
    #             STD_accuracy_list_with_after_psm_values_Horizontal.append(
    #                 np.round(np.std(ACC_level_wise[:, ctr_l, ctr_sv]), decimals=4))
    #             ctr_l = ctr_l + 1
    #     ctr_l = 0
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             if sv == 'age':  # being treated differently because there are two case of psm match in age
    #                 print(" Counter values ", ctr_sv, ctr_l)
    #                 print(sv, name,
    #                       np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    #                       np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 Mean_accuracy_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 STD_accuracy_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 print(" Counter values ", ctr_sv, ctr_l)
    #                 print(sv, name,
    #                       np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4),
    #                       np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    #                 Mean_accuracy_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    #                 STD_accuracy_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    #             else:
    #                 Mean_accuracy_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.mean(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 STD_accuracy_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.std(ACC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #             ctr_l = ctr_l + 1
    #     ctr_sv = ctr_sv + 1
    #
    # Accuracy_plot_Horizontal = pd.DataFrame()
    # Accuracy_plot_Horizontal['Mean_values'] = Mean_accuracy_list_with_after_psm_values_Horizontal
    # Accuracy_plot_Horizontal['Std_values'] = STD_accuracy_list_with_after_psm_values_Horizontal
    # Accuracy_plot_Horizontal['method'] = " "
    # Accuracy_plot_Horizontal['sens_Var_level'] = " "
    # Accuracy_plot_Horizontal['x_ticks'] = " "
    # idx = 0
    # idx_for_level = 0
    # for sv in sens_var:
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             Accuracy_plot_Horizontal.at[idx, 'method'] = "Basic"
    #             Accuracy_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #             Accuracy_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[name]
    #             idx = idx + 1
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             if sv == 'age':  # being treated differently because there are two case of psm match in age
    #                 Accuracy_plot_Horizontal.at[idx, 'method'] = "PSM"
    #                 Accuracy_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #                 Accuracy_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[
    #                                                                               name] + "PSM_Mid-young"
    #                 idx = idx + 1
    #                 Accuracy_plot_Horizontal.at[idx, 'method'] = "PSM1"
    #                 Accuracy_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #                 Accuracy_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[
    #                                                                               name] + "PSM_Mid-old"
    #                 idx = idx + 1
    #             else:
    #                 Accuracy_plot_Horizontal.at[idx, 'method'] = "PSM"
    #                 Accuracy_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #                 Accuracy_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[
    #                                                                               name] + "-PSM"
    #                 idx = idx + 1
    #
    # Accuracy_plot_Horizontal.drop(
    #     Accuracy_plot_Horizontal[Accuracy_plot_Horizontal['Mean_values'] == 0].index,
    #     inplace=True)
    # Accuracy_plot_Horizontal = Accuracy_plot_Horizontal.append(
    #     {'Mean_values': np.round(np.mean(ACC), decimals=4),
    #      'Std_values': np.round(np.std(ACC), decimals=4), 'method': 'Basic', 'sens_Var_level': 'All',
    #      'x_ticks': 'All', }, ignore_index=True)
    # Accuracy_plot_Horizontal['x'] = np.arange(len(Accuracy_plot_Horizontal))
    #
    # mkr_dict = {'Basic': 'o', 'PSM': '^', 'PSM1': '*'}
    # color_dict = {'Sex': 'r', 'Race': 'b', 'age': 'g', 'All': 'orange'}
    # groups = Accuracy_plot_Horizontal.groupby(['sens_Var_level', 'method'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # for name, group in groups:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name, color)
    #     plt.scatter(group.Mean_values, group.x, marker=marker, color=color, linewidths=0,
    #                 s=70)
    #     plt.errorbar(group.Mean_values, group.x, xerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                  fmt='None', capsize=1.5)
    #
    # ax.axhline(y=sens_level_count[0] * 2 - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    # ax.axhline(y=(sens_level_count[0] * 2 + sens_level_count[1] * 2) - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    #
    # ax.legend(handles=patches, ncol=4, prop={'size': 7})
    # ax.set_yticklabels(Accuracy_plot_Horizontal['x_ticks'], ha='right')
    # plt.yticks(Accuracy_plot_Horizontal['x'], Accuracy_plot_Horizontal['x_ticks'])
    # start, end = ax.get_xlim()
    # ax.tick_params(axis='x', which='major', labelsize=10)
    # ax.tick_params(axis='y', which='major', labelsize=6)
    # plt.title(" Accuracy values @ " + str(np.round(threshold[0], decimals=3)))
    # plt.savefig(
    #     fig_saving_dir+'/Non-stacked_Horizontal_Accuracy_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #     bbox_inches='tight')
    # plt.savefig(
    #     fig_saving_dir+'/Non-stacked_Horizontal_Accuracy_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #     bbox_inches='tight')
    # plt.close()
    #
    # """
    # # AUROC plots
    #
    Mean_AUROC_list = []
    STD_AUROC_list = []
    ctr_sv = 0
    for sv in sens_var:
        ctr_l = 0
        for name in sens_var_levels:
            if name.split("_")[0] == sv:
                Mean_AUROC_list.append(
                    np.round(np.mean(AUROC_level_wise[:, ctr_l, ctr_sv]), decimals=2))
                # Mean_AUROC_list_with_after_psm_values.append(
                #     np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
                STD_AUROC_list.append(
                    np.round(np.std(AUROC_level_wise[:, ctr_l, ctr_sv]), decimals=2))
                # STD_AUROC_list_with_after_psm_values.append(
                #     np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
                ctr_l = ctr_l + 1
        ctr_sv = ctr_sv + 1
    # AUROC_to_plot = pd.DataFrame()
    # AUROC_to_plot['Mean_values'] = Mean_AUROC_list
    # AUROC_to_plot['Std_values'] = STD_AUROC_list
    # AUROC_to_plot['method'] = " "
    # AUROC_to_plot['sens_Var_level'] = " "
    # AUROC_to_plot['x_ticks'] = " "
    # idx = 0
    # for j in range(len(sens_var_levels)):
    #     AUROC_to_plot.at[idx, 'method'] = "Basic"
    #     AUROC_to_plot.at[idx, 'sens_Var_level'] = sens_var_levels[j]
    #     AUROC_to_plot.at[idx, 'x_ticks'] = sens_var_level_dict[sens_var_levels[j]]
    #     idx = idx + 1
    # # AUROC_to_plot.drop(AUROC_to_plot[AUROC_to_plot['Mean_values'] == 0].index,
    # #                                inplace=True)
    # AUROC_to_plot['x'] = np.arange(len(AUROC_to_plot))
    # mkr_dict = {'Basic': 'o', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'RACE': 'b', 'age': 'g', 'All': 'orange'}
    # groups = AUROC_to_plot.groupby(['sens_Var_level'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # y_all_mean = np.round(np.mean(AUROC), decimals=2)
    # y_all_std = np.round(np.std(AUROC), decimals=2)
    # ax.axhline(y=y_all_mean, xmin=-1, xmax=1, color='brown', linestyle='--', lw=1,
    #            label="All")
    # # ax.axhline(y=true_outcome_rate, xmin=-1, xmax=1, color='orange', linestyle='--', lw=1,
    # #            label="True_outcome_rate_all")
    # plt.fill_between(np.arange(len(AUROC_to_plot)), y_all_mean - y_all_std, y_all_mean + y_all_std,
    #                  color='lightcyan')
    # for name, group in groups:
    #     print(name)
    #     print(group)
    #     # print(mkr_dict[name[1]])
    #     print(color_dict[name.split("_")[0]])
    #     # marker = mkr_dict[name[1]]
    #     color = color_dict[name.split("_")[0]]
    #     # print(name)
    #     print(" Check point 0")
    #     if name == 'All':
    #         plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none' ,marker='o')
    #         plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                      fmt='None', capsize=1.5)
    #         print("Check point 1")
    #     elif name.split("_")[0] == 'age':
    #         print("Check point 2")
    #         if (name.split("_")[1] != '0.0') & (
    #                 ctr_legend == 0):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values,
    #                         label='(' + str(name.split("_")[0]) + ')', color=color, linewidth=0,
    #                         ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     else:
    #         print("Check point 3")
    #         if ((name.split("_")[1] == str(0.0)) or (name.split("_")[1] == str(1)) or (
    #                 (name.split("_")[1] == str(1.0)))) & (name.split("_")[0] != 'age'):
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values,
    #                         label='(' + str(name.split("_")[0]) + ')', color=color,
    #                         linewidth=0,
    #                         ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #         else:
    #             plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #
    # # Mean_AUROC_list_with_after_psm_values = []
    # # STD_AUROC_list_with_after_psm_values = []
    # # ctr_sv = 0
    # # for sv in sens_var:
    # #     ctr_l = 0
    # #     for name in sens_var_levels:
    # #         if name.split("_")[0] == sv:
    # #             if sv == 'age':  # being treated differently because there are two case of psm match in age
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    # #                       np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 Mean_AUROC_list_with_after_psm_values.append(
    # #                     np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 STD_AUROC_list_with_after_psm_values.append(
    # #                     np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4),
    # #                       np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 Mean_AUROC_list_with_after_psm_values.append(
    # #                     np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 STD_AUROC_list_with_after_psm_values.append(
    # #                     np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #             elif sv == 'Race':  # being treated differently because there are two case of psm match in age
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    # #                       np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 Mean_AUROC_list_with_after_psm_values.append(
    # #                     np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 STD_AUROC_list_with_after_psm_values.append(
    # #                     np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4),
    # #                       np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 Mean_AUROC_list_with_after_psm_values.append(
    # #                     np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 STD_AUROC_list_with_after_psm_values.append(
    # #                     np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4),
    # #                       np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4))
    # #                 Mean_AUROC_list_with_after_psm_values.append(
    # #                     np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4))
    # #                 STD_AUROC_list_with_after_psm_values.append(
    # #                     np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4))
    # #             else:
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    # #                       np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 Mean_AUROC_list_with_after_psm_values.append(
    # #                     np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 STD_AUROC_list_with_after_psm_values.append(
    # #                     np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #             ctr_l = ctr_l + 1
    # #     if sv == 'age':
    # #         ctr_sv = ctr_sv + 2
    # #     if sv == 'Race':
    # #         ctr_sv = ctr_sv + 3
    # #     if sv == 'Sex':
    # #         ctr_sv = ctr_sv + 1
    #
    # AUROC_average_across_bts_level_wise_psm = np.mean(AUROC_level_wise_after_psm, axis=0)
    # AUROC_std_across_bts_level_wise_psm = np.std(AUROC_level_wise_after_psm, axis=0)
    #
    # Mean_AUROC_list_with_after_psm_values = [np.round(AUROC_average_across_bts_level_wise_psm[i,j], decimals=2) for j in range(6) for i in range(3)]
    # STD_AUROC_list_with_after_psm_values = [np.round(AUROC_std_across_bts_level_wise_psm[i,j], decimals=2) for j in range(6) for i in range(3)]
    #
    # Mean_AUROC_list_with_after_psm_values = list(filter(lambda num: num != 0, Mean_AUROC_list_with_after_psm_values))
    # STD_AUROC_list_with_after_psm_values = list(filter(lambda num: num != 0, STD_AUROC_list_with_after_psm_values))
    # AUROC_to_plot_psm = pd.DataFrame()
    # AUROC_to_plot_psm['sens_Var_level_crude'] = Average_risk_score_level_wise_list_after_psm.keys()
    # AUROC_to_plot_psm['Mean_values'] = Mean_AUROC_list_with_after_psm_values
    # AUROC_to_plot_psm['Std_values'] = STD_AUROC_list_with_after_psm_values
    # AUROC_to_plot_psm['method'] = " "
    # AUROC_to_plot_psm['sens_Var_level'] = " "
    # AUROC_to_plot_psm['x_ticks'] = " "
    # AUROC_to_plot_psm['x'] = 0
    # dict_with_levels = dict(zip(sens_var_levels, np.arange(len(sens_var_levels))))
    # idx = 0
    # for j in range(len(Average_risk_score_level_wise_list_after_psm.keys())):
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'Sex':
    #         AUROC_to_plot_psm.at[idx, 'method'] = "PSM"
    #         AUROC_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #             list(Average_risk_score_level_wise_list_after_psm.keys())[j]
    #         AUROC_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[
    #             list(Average_risk_score_level_wise_list_after_psm.keys())[j]]
    #         AUROC_to_plot_psm.at[idx, 'x'] = dict_with_levels[AUROC_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'age':
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[1] == '0psm10':
    #             AUROC_to_plot_psm.at[idx, 'method'] = "PSM"
    #             AUROC_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #             AUROC_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_1-0"
    #             AUROC_to_plot_psm.at[idx, 'x'] = dict_with_levels[AUROC_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[1] == '0psm12':
    #             AUROC_to_plot_psm.at[idx, 'method'] = "PSM1"
    #             AUROC_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #             AUROC_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_1-2"
    #             AUROC_to_plot_psm.at[idx, 'x'] = dict_with_levels[AUROC_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'RACE':
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'BW':
    #             AUROC_to_plot_psm.at[idx, 'method'] = "PSM"
    #             AUROC_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("psm")[0]
    #             AUROC_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split("psm")[0]] + "-PSM_B-W"
    #             AUROC_to_plot_psm.at[idx, 'x'] = dict_with_levels[AUROC_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'OB':
    #             if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[1] == 'nanpsm':
    #                 AUROC_to_plot_psm.at[idx, 'method'] = "PSM1"
    #                 AUROC_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"
    #                 AUROC_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"] + "-PSM_O-B"
    #                 AUROC_to_plot_psm.at[idx, 'x'] = dict_with_levels[AUROC_to_plot_psm['sens_Var_level'][idx]]
    #             else:
    #                 AUROC_to_plot_psm.at[idx, 'method'] = "PSM1"
    #                 AUROC_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #                 AUROC_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_O-B"
    #                 AUROC_to_plot_psm.at[idx, 'x'] = dict_with_levels[AUROC_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'OW':
    #             if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[1] == 'nanpsm':
    #                 AUROC_to_plot_psm.at[idx, 'method'] = "PSM2"
    #                 AUROC_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"
    #                 AUROC_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"] + "-PSM_O-W"
    #                 AUROC_to_plot_psm.at[idx, 'x'] = dict_with_levels[AUROC_to_plot_psm['sens_Var_level'][idx]]
    #             else:
    #                 AUROC_to_plot_psm.at[idx, 'method'] = "PSM2"
    #                 AUROC_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #                 AUROC_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_O-W"
    #                 AUROC_to_plot_psm.at[idx, 'x'] = dict_with_levels[AUROC_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    # AUROC_to_plot_psm['x'] = AUROC_to_plot_psm['x'] + 0.15
    #
    # mkr_dict = {'PSM1': '*', 'PSM': '^', 'PSM2': 's'}
    # color_dict = {'Sex': 'r', 'RACE': 'b', 'age': 'g'}
    # groups_psm = AUROC_to_plot_psm.groupby(['sens_Var_level', 'method'])
    # ctr_legend = 0
    # for name, group in groups_psm:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name)
    #     if name[0] == 'All':
    #         plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, s=5)
    #         plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                      fmt='None', capsize=1.5)
    #     elif name[0].split("_")[0] == 'age':
    #         if (name[0].split("_")[1] == '1.0') & (
    #                 ctr_legend != 2):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Mid-young)'
    #             else:
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Mid-old)'
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label=label_value, color=color, linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     elif name[0].split("_")[0] == 'RACE':
    #         if (name[0].split("_")[
    #             1] == '1'):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Black-White)'
    #             if name[1] == 'PSM1':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Black-Others)'
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label=label_value, color=color, linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         elif (name[0].split("_")[
    #                   1] == '0'):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM2':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_White-Others)'
    #                 plt.plot(group.x, group.Mean_values, marker=marker,
    #                             label=label_value, color=color, linewidth=0,
    #                             ms=8, fillstyle='none')
    #             else:
    #                 plt.plot(group.x, group.Mean_values, marker=marker,
    #                             color=color, linewidth=0,
    #                             ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     else:
    #         if ((name[0].split("_")[1] == str(0.0)) or (name[0].split("_")[1] == str(1)) or (
    #                 (name[0].split("_")[1] == str(1.0)))) & (name[0].split("_")[0] != 'age'):
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label='(' + str(name[0].split("_")[0]) + ',' + str(name[1]) + ')', color=color,
    #                         linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    # ax.axvline(x=sens_level_count[0] - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.axvline(x=(sens_level_count[0] + sens_level_count[
    #     1]) - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.axvline(x=(sens_level_count[0] + sens_level_count[1] + sens_level_count[
    #     2]) - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.legend(ncol=3, prop={'size': 7})
    # ax.set_xticklabels(AUROC_to_plot['x_ticks'])
    # plt.xticks(AUROC_to_plot['x'], AUROC_to_plot['x_ticks'])
    # start, end = ax.get_ylim()
    # ax.tick_params(axis='x', which='major', labelsize=8)
    # ax.tick_params(axis='y', which='major', labelsize=10)
    # ax.yaxis.set_ticks(np.arange(max(0, start - 0.2), min(1.01, end + 0.3), 0.1))
    # plt.xlabel('Sensitive variable levels', fontsize=7)
    # plt.ylabel('AUROC values', fontsize=7)
    # plt.title(" AUROC @ " + str(np.round(threshold[0], decimals=2)))
    # plt.savefig(fig_saving_dir+'/AUROC_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #             bbox_inches='tight')
    # plt.savefig(fig_saving_dir+'/AUROC_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #             bbox_inches='tight')
    # plt.close()
    # ###### alternative plotting tech
    # """
    # AUROC_to_plot = AUROC_to_plot.append(
    #     {'Mean_values': np.round(np.mean(AUROC), decimals=4),
    #      'Std_values': np.round(np.std(AUROC), decimals=4), 'method': 'Basic', 'sens_Var_level': 'All',
    #      'x_ticks': 'All', 'x': len(AUROC_to_plot)}, ignore_index=True)
    #
    # mkr_dict = {'Basic': 'o', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'Race': 'b', 'age': 'g', 'All': 'orange'}
    # groups = AUROC_to_plot.groupby(['sens_Var_level'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # for name, group in groups:
    #     # marker = mkr_dict[name[1]]
    #     color = color_dict[name.split("_")[0]]
    #     # print(name, color)
    #     plt.scatter(group.Mean_values, group.x, color=color, linewidths=0,
    #                 s=70)
    #     plt.errorbar(group.Mean_values, group.x, xerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                  fmt='None', capsize=1.5)
    #
    # mkr_dict = {'PSM1': '*', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'Race': 'b', 'age': 'g'}
    # groups_psm = AUROC_to_plot_psm.groupby(['sens_Var_level', 'method'])
    # # fig, ax = plt.subplots()
    # # ax.margins(0.05)
    # ctr_legend = 0
    # for name, group in groups_psm:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name, color)
    #     plt.scatter(group.Mean_values, group.x, marker=marker, color=color, linewidths=0,
    #                 s=70)
    #     plt.errorbar(group.Mean_values, group.x, xerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                  fmt='None', capsize=1.5)
    #
    # ax.axhline(y=sens_level_count[0] - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    # ax.axhline(y=(sens_level_count[0] + sens_level_count[1]) - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    #
    # ax.legend(handles=patches, ncol=4, prop={'size': 7})
    # ax.set_yticklabels(AUROC_to_plot['x_ticks'], ha='right')
    # plt.yticks(AUROC_to_plot['x'], AUROC_to_plot['x_ticks'])
    # start, end = ax.get_xlim()
    # ax.tick_params(axis='x', which='major', labelsize=10)
    # ax.tick_params(axis='y', which='major', labelsize=10)
    # plt.title(" AUROC values @ " + str(np.round(threshold[0], decimals=3)))
    # plt.savefig(fig_saving_dir+'/Horizontal_AUROC_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #             bbox_inches='tight')
    # plt.savefig(fig_saving_dir+'/Horizontal_AUROC_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #             bbox_inches='tight')
    # plt.close()
    #
    # Mean_AUROC_list_with_after_psm_values_Horizontal = []
    # STD_AUROC_list_with_after_psm_values_Horizontal = []
    # ctr_sv = 0
    # for sv in sens_var:
    #     ctr_l = 0
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             Mean_AUROC_list_with_after_psm_values_Horizontal.append(
    #                 np.round(np.mean(AUROC_level_wise[:, ctr_l, ctr_sv]), decimals=4))
    #             STD_AUROC_list_with_after_psm_values_Horizontal.append(
    #                 np.round(np.std(AUROC_level_wise[:, ctr_l, ctr_sv]), decimals=4))
    #             ctr_l = ctr_l + 1
    #     ctr_l = 0
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             if sv == 'age':  # being treated differently because there are two case of psm match in age
    #                 print(" Counter values ", ctr_sv, ctr_l)
    #                 print(sv, name,
    #                       np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    #                       np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 Mean_AUROC_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 STD_AUROC_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 print(" Counter values ", ctr_sv, ctr_l)
    #                 print(sv, name,
    #                       np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4),
    #                       np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    #                 Mean_AUROC_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    #                 STD_AUROC_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    #             else:
    #                 Mean_AUROC_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 STD_AUROC_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #             ctr_l = ctr_l + 1
    #     ctr_sv = ctr_sv + 1
    #
    # AUROC_plot_Horizontal = pd.DataFrame()
    # AUROC_plot_Horizontal['Mean_values'] = Mean_AUROC_list_with_after_psm_values_Horizontal
    # AUROC_plot_Horizontal['Std_values'] = STD_AUROC_list_with_after_psm_values_Horizontal
    # AUROC_plot_Horizontal['method'] = " "
    # AUROC_plot_Horizontal['sens_Var_level'] = " "
    # AUROC_plot_Horizontal['x_ticks'] = " "
    # idx = 0
    # idx_for_level = 0
    # for sv in sens_var:
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             AUROC_plot_Horizontal.at[idx, 'method'] = "Basic"
    #             AUROC_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #             AUROC_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[name]
    #             idx = idx + 1
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             if sv == 'age':  # being treated differently because there are two case of psm match in age
    #                 AUROC_plot_Horizontal.at[idx, 'method'] = "PSM"
    #                 AUROC_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #                 AUROC_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[
    #                                                                               name] + "PSM_Mid-young"
    #                 idx = idx + 1
    #                 AUROC_plot_Horizontal.at[idx, 'method'] = "PSM1"
    #                 AUROC_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #                 AUROC_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[
    #                                                                               name] + "PSM_Mid-old"
    #                 idx = idx + 1
    #             else:
    #                 AUROC_plot_Horizontal.at[idx, 'method'] = "PSM"
    #                 AUROC_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #                 AUROC_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[
    #                                                                               name] + "-PSM"
    #                 idx = idx + 1
    #
    # AUROC_plot_Horizontal.drop(
    #     AUROC_plot_Horizontal[AUROC_plot_Horizontal['Mean_values'] == 0].index,
    #     inplace=True)
    # AUROC_plot_Horizontal = AUROC_plot_Horizontal.append(
    #     {'Mean_values': np.round(np.mean(AUROC), decimals=4),
    #      'Std_values': np.round(np.std(AUROC), decimals=4), 'method': 'Basic', 'sens_Var_level': 'All',
    #      'x_ticks': 'All', }, ignore_index=True)
    # AUROC_plot_Horizontal['x'] = np.arange(len(AUROC_plot_Horizontal))
    #
    # mkr_dict = {'Basic': 'o', 'PSM': '^', 'PSM1': '*'}
    # color_dict = {'Sex': 'r', 'Race': 'b', 'age': 'g', 'All': 'orange'}
    # groups = AUROC_plot_Horizontal.groupby(['sens_Var_level', 'method'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # for name, group in groups:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name, color)
    #     plt.scatter(group.Mean_values, group.x, marker=marker, color=color, linewidths=0,
    #                 s=70)
    #     plt.errorbar(group.Mean_values, group.x, xerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                  fmt='None', capsize=1.5)
    #
    # ax.axhline(y=sens_level_count[0] * 2 - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    # ax.axhline(y=(sens_level_count[0] * 2 + sens_level_count[1] * 2) - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    #
    # ax.legend(handles=patches, ncol=4, prop={'size': 7})
    # ax.set_yticklabels(AUROC_plot_Horizontal['x_ticks'], ha='right')
    # plt.yticks(AUROC_plot_Horizontal['x'], AUROC_plot_Horizontal['x_ticks'])
    # start, end = ax.get_xlim()
    # ax.tick_params(axis='x', which='major', labelsize=10)
    # ax.tick_params(axis='y', which='major', labelsize=6)
    # plt.title(" AUROC values @ " + str(np.round(threshold[0], decimals=3)))
    # plt.savefig(
    #     fig_saving_dir+'/Non-stacked_Horizontal_AUROC_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #     bbox_inches='tight')
    # plt.savefig(
    #     fig_saving_dir+'/Non-stacked_Horizontal_AUROC_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #     bbox_inches='tight')
    # plt.close()
    # """
    # # PPV plots
    #
    Mean_PPV_list = []
    STD_PPV_list = []
    ctr_sv = 0
    for sv in sens_var:
        ctr_l = 0
        for name in sens_var_levels:
            if name.split("_")[0] == sv:
                Mean_PPV_list.append(
                    np.round(np.mean(PPV_level_wise[:, ctr_l, ctr_sv]), decimals=4))
                # Mean_PPV_list_with_after_psm_values.append(
                #     np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
                STD_PPV_list.append(
                    np.round(np.std(PPV_level_wise[:, ctr_l, ctr_sv]), decimals=4))
                # STD_PPV_list_with_after_psm_values.append(
                #     np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
                ctr_l = ctr_l + 1
        ctr_sv = ctr_sv + 1
    # PPV_to_plot = pd.DataFrame()
    # PPV_to_plot['Mean_values'] = Mean_PPV_list
    # PPV_to_plot['Std_values'] = STD_PPV_list
    # PPV_to_plot['method'] = " "
    # PPV_to_plot['sens_Var_level'] = " "
    # PPV_to_plot['x_ticks'] = " "
    # idx = 0
    # for j in range(len(sens_var_levels)):
    #     PPV_to_plot.at[idx, 'method'] = "Basic"
    #     PPV_to_plot.at[idx, 'sens_Var_level'] = sens_var_levels[j]
    #     PPV_to_plot.at[idx, 'x_ticks'] = sens_var_level_dict[sens_var_levels[j]]
    #     idx = idx + 1
    # # PPV_to_plot.drop(PPV_to_plot[PPV_to_plot['Mean_values'] == 0].index,
    # #                                inplace=True)
    # PPV_to_plot['x'] = np.arange(len(PPV_to_plot))
    # mkr_dict = {'Basic': 'o', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'RACE': 'b', 'age': 'g', 'All': 'orange'}
    # groups = PPV_to_plot.groupby(['sens_Var_level'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # y_all_mean = np.round(np.mean(PPV), decimals=2)
    # y_all_std = np.round(np.std(PPV), decimals=4)
    # ax.axhline(y=y_all_mean, xmin=-1, xmax=1, color='brown', linestyle='--', lw=1,
    #            label="All")
    # # ax.axhline(y=true_outcome_rate, xmin=-1, xmax=1, color='orange', linestyle='--', lw=1,
    # #            label="True_outcome_rate_all")
    # plt.fill_between(np.arange(len(PPV_to_plot)), y_all_mean - y_all_std, y_all_mean + y_all_std,
    #                  color='lightcyan')
    # for name, group in groups:
    #     print(name)
    #     print(group)
    #     # print(mkr_dict[name[1]])
    #     print(color_dict[name.split("_")[0]])
    #     # marker = mkr_dict[name[1]]
    #     color = color_dict[name.split("_")[0]]
    #     # print(name)
    #     print(" Check point 0")
    #     if name == 'All':
    #         plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none' ,marker='o')
    #         plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                      fmt='None', capsize=1.5)
    #         print("Check point 1")
    #     elif name.split("_")[0] == 'age':
    #         print("Check point 2")
    #         if (name.split("_")[1] != '0.0') & (
    #                 ctr_legend == 0):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values,
    #                         label='(' + str(name.split("_")[0]) + ')', color=color, linewidth=0,
    #                         ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     else:
    #         print("Check point 3")
    #         if ((name.split("_")[1] == str(0.0)) or (name.split("_")[1] == str(1)) or (
    #                 (name.split("_")[1] == str(1.0)))) & (name.split("_")[0] != 'age'):
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values,
    #                         label='(' + str(name.split("_")[0]) + ')', color=color,
    #                         linewidth=0,
    #                         ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #         else:
    #             plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #
    # # Mean_PPV_list_with_after_psm_values = []
    # # STD_PPV_list_with_after_psm_values = []
    # # ctr_sv = 0
    # # for sv in sens_var:
    # #     ctr_l = 0
    # #     for name in sens_var_levels:
    # #         if name.split("_")[0] == sv:
    # #             if sv == 'age':  # being treated differently because there are two case of psm match in age
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    # #                       np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 Mean_PPV_list_with_after_psm_values.append(
    # #                     np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 STD_PPV_list_with_after_psm_values.append(
    # #                     np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4),
    # #                       np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 Mean_PPV_list_with_after_psm_values.append(
    # #                     np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 STD_PPV_list_with_after_psm_values.append(
    # #                     np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #             elif sv == 'Race':  # being treated differently because there are two case of psm match in age
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    # #                       np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 Mean_PPV_list_with_after_psm_values.append(
    # #                     np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 STD_PPV_list_with_after_psm_values.append(
    # #                     np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4),
    # #                       np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 Mean_PPV_list_with_after_psm_values.append(
    # #                     np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 STD_PPV_list_with_after_psm_values.append(
    # #                     np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4),
    # #                       np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4))
    # #                 Mean_PPV_list_with_after_psm_values.append(
    # #                     np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4))
    # #                 STD_PPV_list_with_after_psm_values.append(
    # #                     np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4))
    # #             else:
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    # #                       np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 Mean_PPV_list_with_after_psm_values.append(
    # #                     np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 STD_PPV_list_with_after_psm_values.append(
    # #                     np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #             ctr_l = ctr_l + 1
    # #     if sv == 'age':
    # #         ctr_sv = ctr_sv + 2
    # #     if sv == 'Race':
    # #         ctr_sv = ctr_sv + 3
    # #     if sv == 'Sex':
    # #         ctr_sv = ctr_sv + 1
    #
    # PPV_average_across_bts_level_wise_psm = np.mean(PPV_level_wise_after_psm, axis=0)
    # PPV_std_across_bts_level_wise_psm = np.std(PPV_level_wise_after_psm, axis=0)
    #
    # Mean_PPV_list_with_after_psm_values = [np.round(PPV_average_across_bts_level_wise_psm[i,j], decimals=2) for j in range(6) for i in range(3)]
    # STD_PPV_list_with_after_psm_values = [np.round(PPV_std_across_bts_level_wise_psm[i,j], decimals=4) for j in range(6) for i in range(3)]
    #
    #
    # Mean_PPV_list_with_after_psm_values = list(filter(lambda num: num != 0, Mean_PPV_list_with_after_psm_values))
    # STD_PPV_list_with_after_psm_values = list(filter(lambda num: num != 0, STD_PPV_list_with_after_psm_values))
    # PPV_to_plot_psm = pd.DataFrame()
    # PPV_to_plot_psm['sens_Var_level_crude'] = Average_risk_score_level_wise_list_after_psm.keys()
    # PPV_to_plot_psm['Mean_values'] = Mean_PPV_list_with_after_psm_values
    # PPV_to_plot_psm['Std_values'] = STD_PPV_list_with_after_psm_values
    # PPV_to_plot_psm['method'] = " "
    # PPV_to_plot_psm['sens_Var_level'] = " "
    # PPV_to_plot_psm['x_ticks'] = " "
    # PPV_to_plot_psm['x'] = 0
    # dict_with_levels = dict(zip(sens_var_levels, np.arange(len(sens_var_levels))))
    # idx = 0
    # for j in range(len(Average_risk_score_level_wise_list_after_psm.keys())):
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'Sex':
    #         PPV_to_plot_psm.at[idx, 'method'] = "PSM"
    #         PPV_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #             list(Average_risk_score_level_wise_list_after_psm.keys())[j]
    #         PPV_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[
    #             list(Average_risk_score_level_wise_list_after_psm.keys())[j]]
    #         PPV_to_plot_psm.at[idx, 'x'] = dict_with_levels[PPV_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'age':
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[1] == '0psm10':
    #             PPV_to_plot_psm.at[idx, 'method'] = "PSM"
    #             PPV_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #             PPV_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_1-0"
    #             PPV_to_plot_psm.at[idx, 'x'] = dict_with_levels[PPV_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[1] == '0psm12':
    #             PPV_to_plot_psm.at[idx, 'method'] = "PSM1"
    #             PPV_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #             PPV_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_1-2"
    #             PPV_to_plot_psm.at[idx, 'x'] = dict_with_levels[PPV_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'RACE':
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'BW':
    #             PPV_to_plot_psm.at[idx, 'method'] = "PSM"
    #             PPV_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("psm")[0]
    #             PPV_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split("psm")[0]] + "-PSM_B-W"
    #             PPV_to_plot_psm.at[idx, 'x'] = dict_with_levels[PPV_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'OB':
    #             if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[1] == 'nanpsm':
    #                 PPV_to_plot_psm.at[idx, 'method'] = "PSM1"
    #                 PPV_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"
    #                 PPV_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"] + "-PSM_O-B"
    #                 PPV_to_plot_psm.at[idx, 'x'] = dict_with_levels[PPV_to_plot_psm['sens_Var_level'][idx]]
    #             else:
    #                 PPV_to_plot_psm.at[idx, 'method'] = "PSM1"
    #                 PPV_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #                 PPV_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_O-B"
    #                 PPV_to_plot_psm.at[idx, 'x'] = dict_with_levels[PPV_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'OW':
    #             if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[1] == 'nanpsm':
    #                 PPV_to_plot_psm.at[idx, 'method'] = "PSM2"
    #                 PPV_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"
    #                 PPV_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"] + "-PSM_O-W"
    #                 PPV_to_plot_psm.at[idx, 'x'] = dict_with_levels[PPV_to_plot_psm['sens_Var_level'][idx]]
    #             else:
    #                 PPV_to_plot_psm.at[idx, 'method'] = "PSM2"
    #                 PPV_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #                 PPV_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_O-W"
    #                 PPV_to_plot_psm.at[idx, 'x'] = dict_with_levels[PPV_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    # PPV_to_plot_psm['x'] = PPV_to_plot_psm['x'] + 0.15
    #
    # mkr_dict = {'PSM1': '*', 'PSM': '^', 'PSM2': 's'}
    # color_dict = {'Sex': 'r', 'RACE': 'b', 'age': 'g'}
    # groups_psm = PPV_to_plot_psm.groupby(['sens_Var_level', 'method'])
    # ctr_legend = 0
    # for name, group in groups_psm:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name)
    #     if name[0] == 'All':
    #         plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, s=5)
    #         plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                      fmt='None', capsize=1.5)
    #     elif name[0].split("_")[0] == 'age':
    #         if (name[0].split("_")[1] == '1.0') & (
    #                 ctr_legend != 2):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Mid-young)'
    #             else:
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Mid-old)'
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label=label_value, color=color, linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     elif name[0].split("_")[0] == 'RACE':
    #         if (name[0].split("_")[
    #             1] == '1'):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Black-White)'
    #             if name[1] == 'PSM1':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Black-Others)'
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label=label_value, color=color, linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         elif (name[0].split("_")[
    #                   1] == '0'):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM2':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_White-Others)'
    #                 plt.plot(group.x, group.Mean_values, marker=marker,
    #                             label=label_value, color=color, linewidth=0,
    #                             ms=8, fillstyle='none')
    #             else:
    #                 plt.plot(group.x, group.Mean_values, marker=marker,
    #                             color=color, linewidth=0,
    #                             ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     else:
    #         if ((name[0].split("_")[1] == str(0.0)) or (name[0].split("_")[1] == str(1)) or (
    #                 (name[0].split("_")[1] == str(1.0)))) & (name[0].split("_")[0] != 'age'):
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label='(' + str(name[0].split("_")[0]) + ',' + str(name[1]) + ')', color=color,
    #                         linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    # ax.axvline(x=sens_level_count[0] - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.axvline(x=(sens_level_count[0] + sens_level_count[
    #     1]) - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.axvline(x=(sens_level_count[0] + sens_level_count[1] + sens_level_count[
    #     2]) - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.legend(ncol=3, prop={'size': 7})
    # ax.set_xticklabels(PPV_to_plot['x_ticks'])
    # plt.xticks(PPV_to_plot['x'], PPV_to_plot['x_ticks'])
    # start, end = ax.get_ylim()
    # ax.tick_params(axis='x', which='major', labelsize=8)
    # ax.tick_params(axis='y', which='major', labelsize=10)
    # ax.yaxis.set_ticks(np.arange(max(0, start - 0.2), min(1.01, end + 0.3), 0.1))
    # plt.xlabel('Sensitive variable levels', fontsize=7)
    # plt.ylabel('PPV values', fontsize=7)
    # plt.title(" Positive Predictive Value @ " + str(np.round(threshold[0], decimals=2)))
    # plt.savefig(fig_saving_dir+'/PPV_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #             bbox_inches='tight')
    # plt.savefig(fig_saving_dir+'/PPV_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #             bbox_inches='tight')
    # plt.close()
    # ###### alternative plotting tech
    # """
    # PPV_to_plot = PPV_to_plot.append(
    #     {'Mean_values': np.round(np.mean(PPV), decimals=4),
    #      'Std_values': np.round(np.std(PPV), decimals=4), 'method': 'Basic', 'sens_Var_level': 'All',
    #      'x_ticks': 'All', 'x': len(PPV_to_plot)}, ignore_index=True)
    #
    # mkr_dict = {'Basic': 'o', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'Race': 'b', 'age': 'g', 'All': 'orange'}
    # groups = PPV_to_plot.groupby(['sens_Var_level'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # for name, group in groups:
    #     # marker = mkr_dict[name[1]]
    #     color = color_dict[name.split("_")[0]]
    #     # print(name, color)
    #     plt.scatter(group.Mean_values, group.x, color=color, linewidths=0,
    #                 s=70)
    #     plt.errorbar(group.Mean_values, group.x, xerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                  fmt='None', capsize=1.5)
    #
    # mkr_dict = {'PSM1': '*', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'Race': 'b', 'age': 'g'}
    # groups_psm = PPV_to_plot_psm.groupby(['sens_Var_level', 'method'])
    # # fig, ax = plt.subplots()
    # # ax.margins(0.05)
    # ctr_legend = 0
    # for name, group in groups_psm:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name, color)
    #     plt.scatter(group.Mean_values, group.x, marker=marker, color=color, linewidths=0,
    #                 s=70)
    #     plt.errorbar(group.Mean_values, group.x, xerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                  fmt='None', capsize=1.5)
    #
    # ax.axhline(y=sens_level_count[0] - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    # ax.axhline(y=(sens_level_count[0] + sens_level_count[1]) - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    #
    # ax.legend(handles=patches, ncol=4, prop={'size': 7})
    # ax.set_yticklabels(PPV_to_plot['x_ticks'], ha='right')
    # plt.yticks(PPV_to_plot['x'], PPV_to_plot['x_ticks'])
    # start, end = ax.get_xlim()
    # ax.tick_params(axis='x', which='major', labelsize=10)
    # ax.tick_params(axis='y', which='major', labelsize=10)
    # plt.title(" Predictive positive value @ " + str(np.round(threshold[0], decimals=3)))
    # plt.savefig(fig_saving_dir+'/Horizontal_PPV_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #             bbox_inches='tight')
    # plt.savefig(fig_saving_dir+'/Horizontal_PPV_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #             bbox_inches='tight')
    # plt.close()
    #
    # Mean_PPV_list_with_after_psm_values_Horizontal = []
    # STD_PPV_list_with_after_psm_values_Horizontal = []
    # ctr_sv = 0
    # for sv in sens_var:
    #     ctr_l = 0
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             Mean_PPV_list_with_after_psm_values_Horizontal.append(
    #                 np.round(np.mean(PPV_level_wise[:, ctr_l, ctr_sv]), decimals=4))
    #             STD_PPV_list_with_after_psm_values_Horizontal.append(
    #                 np.round(np.std(PPV_level_wise[:, ctr_l, ctr_sv]), decimals=4))
    #             ctr_l = ctr_l + 1
    #     ctr_l = 0
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             if sv == 'age':  # being treated differently because there are two case of psm match in age
    #                 print(" Counter values ", ctr_sv, ctr_l)
    #                 print(sv, name,
    #                       np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    #                       np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 Mean_PPV_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 STD_PPV_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 print(" Counter values ", ctr_sv, ctr_l)
    #                 print(sv, name,
    #                       np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4),
    #                       np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    #                 Mean_PPV_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    #                 STD_PPV_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    #             else:
    #                 Mean_PPV_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 STD_PPV_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #             ctr_l = ctr_l + 1
    #     ctr_sv = ctr_sv + 1
    #
    # PPV_plot_Horizontal = pd.DataFrame()
    # PPV_plot_Horizontal['Mean_values'] = Mean_PPV_list_with_after_psm_values_Horizontal
    # PPV_plot_Horizontal['Std_values'] = STD_PPV_list_with_after_psm_values_Horizontal
    # PPV_plot_Horizontal['method'] = " "
    # PPV_plot_Horizontal['sens_Var_level'] = " "
    # PPV_plot_Horizontal['x_ticks'] = " "
    # idx = 0
    # idx_for_level = 0
    # for sv in sens_var:
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             PPV_plot_Horizontal.at[idx, 'method'] = "Basic"
    #             PPV_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #             PPV_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[name]
    #             idx = idx + 1
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             if sv == 'age':  # being treated differently because there are two case of psm match in age
    #                 PPV_plot_Horizontal.at[idx, 'method'] = "PSM"
    #                 PPV_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #                 PPV_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[
    #                                                                               name] + "PSM_Mid-young"
    #                 idx = idx + 1
    #                 PPV_plot_Horizontal.at[idx, 'method'] = "PSM1"
    #                 PPV_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #                 PPV_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[
    #                                                                               name] + "PSM_Mid-old"
    #                 idx = idx + 1
    #             else:
    #                 PPV_plot_Horizontal.at[idx, 'method'] = "PSM"
    #                 PPV_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #                 PPV_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[
    #                                                                               name] + "-PSM"
    #                 idx = idx + 1
    #
    # PPV_plot_Horizontal.drop(
    #     PPV_plot_Horizontal[PPV_plot_Horizontal['Mean_values'] == 0].index,
    #     inplace=True)
    # PPV_plot_Horizontal = PPV_plot_Horizontal.append(
    #     {'Mean_values': np.round(np.mean(PPV), decimals=4),
    #      'Std_values': np.round(np.std(PPV), decimals=4), 'method': 'Basic', 'sens_Var_level': 'All',
    #      'x_ticks': 'All', }, ignore_index=True)
    # PPV_plot_Horizontal['x'] = np.arange(len(PPV_plot_Horizontal))
    #
    # mkr_dict = {'Basic': 'o', 'PSM': '^', 'PSM1': '*'}
    # color_dict = {'Sex': 'r', 'Race': 'b', 'age': 'g', 'All': 'orange'}
    # groups = PPV_plot_Horizontal.groupby(['sens_Var_level', 'method'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # for name, group in groups:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name, color)
    #     plt.scatter(group.Mean_values, group.x, marker=marker, color=color, linewidths=0,
    #                 s=70)
    #     plt.errorbar(group.Mean_values, group.x, xerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                  fmt='None', capsize=1.5)
    #
    # ax.axhline(y=sens_level_count[0] * 2 - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    # ax.axhline(y=(sens_level_count[0] * 2 + sens_level_count[1] * 2) - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    #
    # ax.legend(handles=patches, ncol=4, prop={'size': 7})
    # ax.set_yticklabels(PPV_plot_Horizontal['x_ticks'], ha='right')
    # plt.yticks(PPV_plot_Horizontal['x'], PPV_plot_Horizontal['x_ticks'])
    # start, end = ax.get_xlim()
    # ax.tick_params(axis='x', which='major', labelsize=10)
    # ax.tick_params(axis='y', which='major', labelsize=6)
    # plt.title(" Predictive positive value @ " + str(np.round(threshold[0], decimals=3)))
    # plt.savefig(
    #     fig_saving_dir+'/Non-stacked_Horizontal_PPV_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #     bbox_inches='tight')
    # plt.savefig(
    #     fig_saving_dir+'/Non-stacked_Horizontal_PPV_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #     bbox_inches='tight')
    # plt.close()
    #
    # """
    #
    # #SensitivityPlots
    #
    Mean_Sensitivity_list = []
    STD_Sensitivity_list = []
    ctr_sv = 0
    for sv in sens_var:
        ctr_l = 0
        for name in sens_var_levels:
            if name.split("_")[0] == sv:
                Mean_Sensitivity_list.append(
                    np.round(np.mean(Sensitivity_level_wise[:, ctr_l, ctr_sv]), decimals=2))
                # Mean_Sensitivity_list_with_after_psm_values.append(
                #     np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
                STD_Sensitivity_list.append(
                    np.round(np.std(Sensitivity_level_wise[:, ctr_l, ctr_sv]), decimals=4))
                # STD_Sensitivity_list_with_after_psm_values.append(
                #     np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
                ctr_l = ctr_l + 1
        ctr_sv = ctr_sv + 1
    # Sensitivity_to_plot = pd.DataFrame()
    # Sensitivity_to_plot['Mean_values'] = Mean_Sensitivity_list
    # Sensitivity_to_plot['Std_values'] = STD_Sensitivity_list
    # Sensitivity_to_plot['method'] = " "
    # Sensitivity_to_plot['sens_Var_level'] = " "
    # Sensitivity_to_plot['x_ticks'] = " "
    # idx = 0
    # for j in range(len(sens_var_levels)):
    #     Sensitivity_to_plot.at[idx, 'method'] = "Basic"
    #     Sensitivity_to_plot.at[idx, 'sens_Var_level'] = sens_var_levels[j]
    #     Sensitivity_to_plot.at[idx, 'x_ticks'] = sens_var_level_dict[sens_var_levels[j]]
    #     idx = idx + 1
    # # Sensitivity_to_plot.drop(Sensitivity_to_plot[Sensitivity_to_plot['Mean_values'] == 0].index,
    # #                                inplace=True)
    # Sensitivity_to_plot['x'] = np.arange(len(Sensitivity_to_plot))
    # mkr_dict = {'Basic': 'o', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'RACE': 'b', 'age': 'g', 'All': 'orange'}
    # groups = Sensitivity_to_plot.groupby(['sens_Var_level'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # y_all_mean = np.round(np.mean(Sensitivity), decimals=2)
    # y_all_std = np.round(np.std(Sensitivity), decimals=4)
    # ax.axhline(y=y_all_mean, xmin=-1, xmax=1, color='brown', linestyle='--', lw=1,
    #            label="All")
    # # ax.axhline(y=true_outcome_rate, xmin=-1, xmax=1, color='orange', linestyle='--', lw=1,
    # #            label="True_outcome_rate_all")
    # plt.fill_between(np.arange(len(Sensitivity_to_plot)), y_all_mean - y_all_std, y_all_mean + y_all_std,
    #                  color='lightcyan')
    # for name, group in groups:
    #     print(name)
    #     print(group)
    #     # print(mkr_dict[name[1]])
    #     print(color_dict[name.split("_")[0]])
    #     # marker = mkr_dict[name[1]]
    #     color = color_dict[name.split("_")[0]]
    #     # print(name)
    #     print(" Check point 0")
    #     if name == 'All':
    #         plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none' ,marker='o')
    #         plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                      fmt='None', capsize=1.5)
    #         print("Check point 1")
    #     elif name.split("_")[0] == 'age':
    #         print("Check point 2")
    #         if (name.split("_")[1] != '0.0') & (
    #                 ctr_legend == 0):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values,
    #                         label='(' + str(name.split("_")[0]) + ')', color=color, linewidth=0,
    #                         ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     else:
    #         print("Check point 3")
    #         if ((name.split("_")[1] == str(0.0)) or (name.split("_")[1] == str(1)) or (
    #                 (name.split("_")[1] == str(1.0)))) & (name.split("_")[0] != 'age'):
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values,
    #                         label='(' + str(name.split("_")[0]) + ')', color=color,
    #                         linewidth=0,
    #                         ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #         else:
    #             plt.plot(group.x, group.Mean_values, color=color, linewidth=0, ms=8, fillstyle='none' ,marker='o')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #
    # # Mean_Sensitivity_list_with_after_psm_values = []
    # # STD_Sensitivity_list_with_after_psm_values = []
    # # ctr_sv = 0
    # # for sv in sens_var:
    # #     ctr_l = 0
    # #     for name in sens_var_levels:
    # #         if name.split("_")[0] == sv:
    # #             if sv == 'age':  # being treated differently because there are two case of psm match in age
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    # #                       np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 Mean_Sensitivity_list_with_after_psm_values.append(
    # #                     np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 STD_Sensitivity_list_with_after_psm_values.append(
    # #                     np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4),
    # #                       np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 Mean_Sensitivity_list_with_after_psm_values.append(
    # #                     np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 STD_Sensitivity_list_with_after_psm_values.append(
    # #                     np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #             elif sv == 'Race':  # being treated differently because there are two case of psm match in age
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    # #                       np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 Mean_Sensitivity_list_with_after_psm_values.append(
    # #                     np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 STD_Sensitivity_list_with_after_psm_values.append(
    # #                     np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4),
    # #                       np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 Mean_Sensitivity_list_with_after_psm_values.append(
    # #                     np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 STD_Sensitivity_list_with_after_psm_values.append(
    # #                     np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4),
    # #                       np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4))
    # #                 Mean_Sensitivity_list_with_after_psm_values.append(
    # #                     np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4))
    # #                 STD_Sensitivity_list_with_after_psm_values.append(
    # #                     np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 2]), decimals=4))
    # #             else:
    # #                 print(" Counter values ", ctr_sv, ctr_l)
    # #                 print(sv, name,
    # #                       np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    # #                       np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 Mean_Sensitivity_list_with_after_psm_values.append(
    # #                     np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #                 STD_Sensitivity_list_with_after_psm_values.append(
    # #                     np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    # #             ctr_l = ctr_l + 1
    # #     if sv == 'age':
    # #         ctr_sv = ctr_sv + 2
    # #     if sv == 'Race':
    # #         ctr_sv = ctr_sv + 3
    # #     if sv == 'Sex':
    # #         ctr_sv = ctr_sv + 1
    #
    # Sensitivity_average_across_bts_level_wise_psm = np.mean(Sensitivity_level_wise_after_psm, axis=0)
    # Sensitivity_std_across_bts_level_wise_psm = np.std(Sensitivity_level_wise_after_psm, axis=0)
    #
    # Mean_Sensitivity_list_with_after_psm_values = [np.round(Sensitivity_average_across_bts_level_wise_psm[i,j], decimals=2) for j in range(6) for i in range(3)]
    # STD_Sensitivity_list_with_after_psm_values = [np.round(Sensitivity_std_across_bts_level_wise_psm[i,j], decimals=4) for j in range(6) for i in range(3)]
    #
    # for i in range(len(Mean_Sensitivity_list_with_after_psm_values)):
    #     if Mean_Sensitivity_list_with_after_psm_values[i]==1.0:
    #         STD_Sensitivity_list_with_after_psm_values[9] = 0.0001
    #
    # Mean_Sensitivity_list_with_after_psm_values = list(
    #     filter(lambda num: num != 0, Mean_Sensitivity_list_with_after_psm_values))
    # STD_Sensitivity_list_with_after_psm_values = list(
    #     filter(lambda num: num != 0, STD_Sensitivity_list_with_after_psm_values))
    # Sensitivity_to_plot_psm = pd.DataFrame()
    # Sensitivity_to_plot_psm['sens_Var_level_crude'] = Average_risk_score_level_wise_list_after_psm.keys()
    # Sensitivity_to_plot_psm['Mean_values'] = Mean_Sensitivity_list_with_after_psm_values
    # Sensitivity_to_plot_psm['Std_values'] = STD_Sensitivity_list_with_after_psm_values
    # Sensitivity_to_plot_psm['method'] = " "
    # Sensitivity_to_plot_psm['sens_Var_level'] = " "
    # Sensitivity_to_plot_psm['x_ticks'] = " "
    # Sensitivity_to_plot_psm['x'] = 0
    # dict_with_levels = dict(zip(sens_var_levels, np.arange(len(sens_var_levels))))
    # idx = 0
    # for j in range(len(Average_risk_score_level_wise_list_after_psm.keys())):
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'Sex':
    #         Sensitivity_to_plot_psm.at[idx, 'method'] = "PSM"
    #         Sensitivity_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #             list(Average_risk_score_level_wise_list_after_psm.keys())[j]
    #         Sensitivity_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[
    #             list(Average_risk_score_level_wise_list_after_psm.keys())[j]]
    #         Sensitivity_to_plot_psm.at[idx, 'x'] = dict_with_levels[Sensitivity_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'age':
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[1] == '0psm10':
    #             Sensitivity_to_plot_psm.at[idx, 'method'] = "PSM"
    #             Sensitivity_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #             Sensitivity_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_1-0"
    #             Sensitivity_to_plot_psm.at[idx, 'x'] = dict_with_levels[Sensitivity_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[1] == '0psm12':
    #             Sensitivity_to_plot_psm.at[idx, 'method'] = "PSM1"
    #             Sensitivity_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #             Sensitivity_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_1-2"
    #             Sensitivity_to_plot_psm.at[idx, 'x'] = dict_with_levels[Sensitivity_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    #     if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] == 'RACE':
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'BW':
    #             Sensitivity_to_plot_psm.at[idx, 'method'] = "PSM"
    #             Sensitivity_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                 list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("psm")[0]
    #             Sensitivity_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                 Average_risk_score_level_wise_list_after_psm.keys())[j].split("psm")[0]] + "-PSM_B-W"
    #             Sensitivity_to_plot_psm.at[idx, 'x'] = dict_with_levels[Sensitivity_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'OB':
    #             if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[1] == 'nanpsm':
    #                 Sensitivity_to_plot_psm.at[idx, 'method'] = "PSM1"
    #                 Sensitivity_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"
    #                 Sensitivity_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"] + "-PSM_O-B"
    #                 Sensitivity_to_plot_psm.at[idx, 'x'] = dict_with_levels[
    #                     Sensitivity_to_plot_psm['sens_Var_level'][idx]]
    #             else:
    #                 Sensitivity_to_plot_psm.at[idx, 'method'] = "PSM1"
    #                 Sensitivity_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #                 Sensitivity_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_O-B"
    #                 Sensitivity_to_plot_psm.at[idx, 'x'] = dict_with_levels[
    #                     Sensitivity_to_plot_psm['sens_Var_level'][idx]]
    #         if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[2] == 'OW':
    #             if list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[1] == 'nanpsm':
    #                 Sensitivity_to_plot_psm.at[idx, 'method'] = "PSM2"
    #                 Sensitivity_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"
    #                 Sensitivity_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split("_")[0] + "_nan"] + "-PSM_O-W"
    #                 Sensitivity_to_plot_psm.at[idx, 'x'] = dict_with_levels[
    #                     Sensitivity_to_plot_psm['sens_Var_level'][idx]]
    #             else:
    #                 Sensitivity_to_plot_psm.at[idx, 'method'] = "PSM2"
    #                 Sensitivity_to_plot_psm.at[idx, 'sens_Var_level'] = \
    #                     list(Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"
    #                 Sensitivity_to_plot_psm.at[idx, 'x_ticks'] = sens_var_level_dict[list(
    #                     Average_risk_score_level_wise_list_after_psm.keys())[j].split(".")[0] + ".0"] + "-PSM_O-W"
    #                 Sensitivity_to_plot_psm.at[idx, 'x'] = dict_with_levels[
    #                     Sensitivity_to_plot_psm['sens_Var_level'][idx]]
    #         idx = idx + 1
    # Sensitivity_to_plot_psm['x'] = Sensitivity_to_plot_psm['x'] + 0.15
    #
    # mkr_dict = {'PSM1': '*', 'PSM': '^', 'PSM2': 's'}
    # color_dict = {'Sex': 'r', 'RACE': 'b', 'age': 'g'}
    # groups_psm = Sensitivity_to_plot_psm.groupby(['sens_Var_level', 'method'])
    # ctr_legend = 0
    # for name, group in groups_psm:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name)
    #     if name[0] == 'All':
    #         plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, s=5)
    #         plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                      fmt='None', capsize=1.5)
    #     elif name[0].split("_")[0] == 'age':
    #         if (name[0].split("_")[1] == '1.0') & (
    #                 ctr_legend != 2):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Mid-young)'
    #             else:
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Mid-old)'
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label=label_value, color=color, linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     elif name[0].split("_")[0] == 'RACE':
    #         if (name[0].split("_")[
    #             1] == '1'):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Black-White)'
    #             if name[1] == 'PSM1':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_Black-Others)'
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label=label_value, color=color, linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         elif (name[0].split("_")[
    #                   1] == '0'):  # there is some commonality between the indexing of age an se, i.e., 2 hence keeping seperate loops
    #             # print("hello")
    #             if name[1] == 'PSM2':
    #                 label_value = '(' + str(name[0].split("_")[0]) + ', PSM_White-Others)'
    #                 plt.plot(group.x, group.Mean_values, marker=marker,
    #                             label=label_value, color=color, linewidth=0,
    #                             ms=8, fillstyle='none')
    #             else:
    #                 plt.plot(group.x, group.Mean_values, marker=marker,
    #                             color=color, linewidth=0,
    #                             ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #             ctr_legend = ctr_legend + 1
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                          fmt='None', capsize=1.5)
    #     else:
    #         if ((name[0].split("_")[1] == str(0.0)) or (name[0].split("_")[1] == str(1)) or (
    #                 (name[0].split("_")[1] == str(1.0)))) & (name[0].split("_")[0] != 'age'):
    #             # print("hello")
    #             plt.plot(group.x, group.Mean_values, marker=marker,
    #                         label='(' + str(name[0].split("_")[0]) + ',' + str(name[1]) + ')', color=color,
    #                         linewidth=0,
    #                         ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    #         else:
    #             plt.plot(group.x, group.Mean_values, marker=marker, color=color, linewidth=0, ms=8, fillstyle='none')
    #             plt.errorbar(group.x, group.Mean_values, yerr=group.Std_values, ecolor='k', elinewidth=0.5, fmt='None',
    #                          capsize=1.5)
    # ax.axvline(x=sens_level_count[0] - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.axvline(x=(sens_level_count[0] + sens_level_count[
    #     1]) - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.axvline(x=(sens_level_count[0] + sens_level_count[1] + sens_level_count[
    #     2]) - 1 + 0.5)  # -1 because the actual x axis is from 0 while plotting
    # ax.legend(ncol=3, prop={'size': 7})
    # ax.set_xticklabels(Sensitivity_to_plot['x_ticks'])
    # plt.xticks(Sensitivity_to_plot['x'], Sensitivity_to_plot['x_ticks'])
    # start, end = ax.get_ylim()
    # ax.tick_params(axis='x', which='major', labelsize=8)
    # ax.tick_params(axis='y', which='major', labelsize=10)
    # ax.yaxis.set_ticks(np.arange(max(0, start - 0.2), min(1.01, end + 0.3), 0.1))
    # plt.xlabel('Sensitive variable levels', fontsize=7)
    # plt.ylabel('Sensitivity values', fontsize=7)
    # plt.title(" Sensitivity @ " + str(np.round(threshold[0], decimals=2)))
    # plt.savefig(fig_saving_dir+'/Sensitivity_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #             bbox_inches='tight')
    # plt.savefig(fig_saving_dir+'/Sensitivity_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #             bbox_inches='tight')
    # plt.close()
    # ###### alternative plotting tech
    # """
    # Sensitivity_to_plot = Sensitivity_to_plot.append(
    #     {'Mean_values': np.round(np.mean(Sensitivity), decimals=4),
    #      'Std_values': np.round(np.std(Sensitivity), decimals=4), 'method': 'Basic', 'sens_Var_level': 'All',
    #      'x_ticks': 'All', 'x': len(Sensitivity_to_plot)}, ignore_index=True)
    #
    # mkr_dict = {'Basic': 'o', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'Race': 'b', 'age': 'g', 'All': 'orange'}
    # groups = Sensitivity_to_plot.groupby(['sens_Var_level'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # for name, group in groups:
    #     # marker = mkr_dict[name[1]]
    #     color = color_dict[name.split("_")[0]]
    #     # print(name, color)
    #     plt.scatter(group.Mean_values, group.x, color=color, linewidths=0,
    #                 s=70)
    #     plt.errorbar(group.Mean_values, group.x, xerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                  fmt='None', capsize=1.5)
    #
    # mkr_dict = {'PSM1': '*', 'PSM': '^'}
    # color_dict = {'Sex': 'r', 'Race': 'b', 'age': 'g'}
    # groups_psm = Sensitivity_to_plot_psm.groupby(['sens_Var_level', 'method'])
    # # fig, ax = plt.subplots()
    # # ax.margins(0.05)
    # ctr_legend = 0
    # for name, group in groups_psm:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name, color)
    #     plt.scatter(group.Mean_values, group.x, marker=marker, color=color, linewidths=0,
    #                 s=70)
    #     plt.errorbar(group.Mean_values, group.x, xerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                  fmt='None', capsize=1.5)
    #
    # ax.axhline(y=sens_level_count[0] - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    # ax.axhline(y=(sens_level_count[0] + sens_level_count[1]) - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    #
    # ax.legend(handles=patches, ncol=4, prop={'size': 7})
    # ax.set_yticklabels(Sensitivity_to_plot['x_ticks'], ha='right')
    # plt.yticks(Sensitivity_to_plot['x'], Sensitivity_to_plot['x_ticks'])
    # start, end = ax.get_xlim()
    # ax.tick_params(axis='x', which='major', labelsize=10)
    # ax.tick_params(axis='y', which='major', labelsize=10)
    # plt.title(" Sensitivity values @ " + str(np.round(threshold[0], decimals=3)))
    # plt.savefig(fig_saving_dir+'/Horizontal_Sensitivity_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #             bbox_inches='tight')
    # plt.savefig(fig_saving_dir+'/Horizontal_Sensitivity_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #             bbox_inches='tight')
    # plt.close()
    #
    # Mean_Sensitivity_list_with_after_psm_values_Horizontal = []
    # STD_Sensitivity_list_with_after_psm_values_Horizontal = []
    # ctr_sv = 0
    # for sv in sens_var:
    #     ctr_l = 0
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             Mean_Sensitivity_list_with_after_psm_values_Horizontal.append(
    #                 np.round(np.mean(Sensitivity_level_wise[:, ctr_l, ctr_sv]), decimals=4))
    #             STD_Sensitivity_list_with_after_psm_values_Horizontal.append(
    #                 np.round(np.std(Sensitivity_level_wise[:, ctr_l, ctr_sv]), decimals=4))
    #             ctr_l = ctr_l + 1
    #     ctr_l = 0
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             if sv == 'age':  # being treated differently because there are two case of psm match in age
    #                 print(" Counter values ", ctr_sv, ctr_l)
    #                 print(sv, name,
    #                       np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4),
    #                       np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 Mean_Sensitivity_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 STD_Sensitivity_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 print(" Counter values ", ctr_sv, ctr_l)
    #                 print(sv, name,
    #                       np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4),
    #                       np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    #                 Mean_Sensitivity_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    #                 STD_Sensitivity_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv + 1]), decimals=4))
    #             else:
    #                 Mean_Sensitivity_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #                 STD_Sensitivity_list_with_after_psm_values_Horizontal.append(
    #                     np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
    #             ctr_l = ctr_l + 1
    #     ctr_sv = ctr_sv + 1
    #
    # Sensitivity_plot_Horizontal = pd.DataFrame()
    # Sensitivity_plot_Horizontal['Mean_values'] = Mean_Sensitivity_list_with_after_psm_values_Horizontal
    # Sensitivity_plot_Horizontal['Std_values'] = STD_Sensitivity_list_with_after_psm_values_Horizontal
    # Sensitivity_plot_Horizontal['method'] = " "
    # Sensitivity_plot_Horizontal['sens_Var_level'] = " "
    # Sensitivity_plot_Horizontal['x_ticks'] = " "
    # idx = 0
    # idx_for_level = 0
    # for sv in sens_var:
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             Sensitivity_plot_Horizontal.at[idx, 'method'] = "Basic"
    #             Sensitivity_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #             Sensitivity_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[name]
    #             idx = idx + 1
    #     for name in sens_var_levels:
    #         if name.split("_")[0] == sv:
    #             if sv == 'age':  # being treated differently because there are two case of psm match in age
    #                 Sensitivity_plot_Horizontal.at[idx, 'method'] = "PSM"
    #                 Sensitivity_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #                 Sensitivity_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[
    #                                                                               name] + "PSM_Mid-young"
    #                 idx = idx + 1
    #                 Sensitivity_plot_Horizontal.at[idx, 'method'] = "PSM1"
    #                 Sensitivity_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #                 Sensitivity_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[
    #                                                                               name] + "PSM_Mid-old"
    #                 idx = idx + 1
    #             else:
    #                 Sensitivity_plot_Horizontal.at[idx, 'method'] = "PSM"
    #                 Sensitivity_plot_Horizontal.at[idx, 'sens_Var_level'] = name
    #                 Sensitivity_plot_Horizontal.at[idx, 'x_ticks'] = sens_var_level_dict[
    #                                                                               name] + "-PSM"
    #                 idx = idx + 1
    #
    # Sensitivity_plot_Horizontal.drop(
    #     Sensitivity_plot_Horizontal[Sensitivity_plot_Horizontal['Mean_values'] == 0].index,
    #     inplace=True)
    # Sensitivity_plot_Horizontal = Sensitivity_plot_Horizontal.append(
    #     {'Mean_values': np.round(np.mean(Sensitivity), decimals=4),
    #      'Std_values': np.round(np.std(Sensitivity), decimals=4), 'method': 'Basic', 'sens_Var_level': 'All',
    #      'x_ticks': 'All', }, ignore_index=True)
    # Sensitivity_plot_Horizontal['x'] = np.arange(len(Sensitivity_plot_Horizontal))
    #
    # mkr_dict = {'Basic': 'o', 'PSM': '^', 'PSM1': '*'}
    # color_dict = {'Sex': 'r', 'Race': 'b', 'age': 'g', 'All': 'orange'}
    # groups = Sensitivity_plot_Horizontal.groupby(['sens_Var_level', 'method'])
    # fig, ax = plt.subplots()
    # ax.margins(0.05)
    # ctr_legend = 0
    # for name, group in groups:
    #     marker = mkr_dict[name[1]]
    #     color = color_dict[name[0].split("_")[0]]
    #     # print(name, color)
    #     plt.scatter(group.Mean_values, group.x, marker=marker, color=color, linewidths=0,
    #                 s=70)
    #     plt.errorbar(group.Mean_values, group.x, xerr=group.Std_values, ecolor='k', elinewidth=0.5,
    #                  fmt='None', capsize=1.5)
    #
    # ax.axhline(y=sens_level_count[0] * 2 - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    # ax.axhline(y=(sens_level_count[0] * 2 + sens_level_count[1] * 2) - 1 + 0.5,
    #            color='brown')  # -1 because the actual y axis is from 0 while plotting
    #
    # ax.legend(handles=patches, ncol=4, prop={'size': 7})
    # ax.set_yticklabels(Sensitivity_plot_Horizontal['x_ticks'], ha='right')
    # plt.yticks(Sensitivity_plot_Horizontal['x'], Sensitivity_plot_Horizontal['x_ticks'])
    # start, end = ax.get_xlim()
    # ax.tick_params(axis='x', which='major', labelsize=10)
    # ax.tick_params(axis='y', which='major', labelsize=6)
    # plt.title(" Sensitivity values @ " + str(np.round(threshold[0], decimals=3)))
    # plt.savefig(
    #     fig_saving_dir+'/Non-stacked_Horizontal_Sensitivity_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.pdf',
    #     bbox_inches='tight')
    # plt.savefig(
    #     fig_saving_dir+'/Non-stacked_Horizontal_Sensitivity_final_fairness_model_reportcard_with_' + str(num_of_bts) + '_BTS.png',
    #     bbox_inches='tight')
    # plt.close()
    #
    # print("Meh")
    # """
    # """


    # Getting the table content

    """ Getting the data in tabular form """

    PPV_Sens_group_averages = np.zeros((num_of_bts, 3))
    PPV_Diff_from_average = np.zeros((num_of_bts, 3, 3))
    for i in range(num_of_bts):
        for j in range(3):  # 3 because there are 3 sensitive attributes
            PPV_Sens_group_averages[i, j] = np.sum(PPV_level_wise[i, :, j]) / np.count_nonzero(PPV_level_wise[i, :, j])
            for k in range(3):
                if PPV_level_wise[i, k, j] != 0:
                    PPV_Diff_from_average[i, k, j] = PPV_level_wise[i, k, j] - PPV_Sens_group_averages[i, j]
    PPV_Sens_group_averages_across_bts = np.mean(PPV_Sens_group_averages, axis=0)

    Mean_Diff_PPV_list = []
    STD_Diff_PPV_list = []
    ctr_sv = 0
    for sv in sens_var:
        ctr_l = 0
        for name in sens_var_levels:
            if name.split("_")[0] == sv:
                print(sv, name,  np.mean(PPV_Diff_from_average[:, ctr_l, ctr_sv]) )
                Mean_Diff_PPV_list.append(
                    np.round(np.mean(PPV_Diff_from_average[:, ctr_l, ctr_sv]), decimals=2))
                # Mean_PPV_list_with_after_psm_values.append(
                #     np.round(np.mean(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
                STD_Diff_PPV_list.append(
                    np.round(np.std(PPV_Diff_from_average[:, ctr_l, ctr_sv]), decimals=4))
                # STD_PPV_list_with_after_psm_values.append(
                #     np.round(np.std(PPV_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
                ctr_l = ctr_l + 1
        ctr_sv = ctr_sv + 1
    # computing the t statistic to test significance in the difference of mean fomr average (single statistic)
    t_stat_PPV_diffFromMean_level_wise = [Mean_Diff_PPV_list[i] / (STD_Diff_PPV_list[i] / np.sqrt(num_of_bts)) for i in
                                          range(len(Mean_Diff_PPV_list))]
    p_value_PPV_diffFromMean_level_wise = [scipy.stats.t.sf(np.abs(x), num_of_bts - 1) for x in
                                           t_stat_PPV_diffFromMean_level_wise]

    PPV_Sens_group_averages_psm = np.zeros((num_of_bts, 6))
    PPV_Diff_from_average_psm = np.zeros((num_of_bts, 3, 6))
    for i in range(num_of_bts):
        for j in range(6):  # 6 because there are 6 psm groups
            PPV_Sens_group_averages_psm[i, j] = np.sum(
                PPV_level_wise_after_psm[i, :, j]) / 2  # because in psm we are matching 2 levels at a time
            for k in range(3):
                if PPV_level_wise_after_psm[i, k, j] != 0:
                    PPV_Diff_from_average_psm[i, k, j] = PPV_level_wise_after_psm[i, k, j] - \
                                                         PPV_Sens_group_averages_psm[i, j]
    PPV_Sens_group_averages_across_bts_psm = np.mean(PPV_Sens_group_averages_psm, axis=0)

    PPV_Diff_from_average_across_bts_level_wise_psm = np.mean(PPV_Diff_from_average_psm, axis=0)
    PPV_Diff_from_std_across_bts_level_wise_psm = np.std(PPV_Diff_from_average_psm, axis=0)

    Mean_Diff_PPV_list_with_after_psm_values = [np.round(PPV_Diff_from_average_across_bts_level_wise_psm[i,j], decimals=6) for j in range(6) for i in range(3)]
    STD_Diff_PPV_list_with_after_psm_values = [np.round(PPV_Diff_from_std_across_bts_level_wise_psm[i,j], decimals=4) for j in range(6) for i in range(3)]


    Mean_Diff_PPV_list_with_after_psm_values = list(
        filter(lambda num: num != 0, Mean_Diff_PPV_list_with_after_psm_values))
    STD_Diff_PPV_list_with_after_psm_values = list(
        filter(lambda num: num != 0, STD_Diff_PPV_list_with_after_psm_values))
    t_stat_PPV_diffFromMean_level_wise_psm = [
        Mean_Diff_PPV_list_with_after_psm_values[i] / (STD_Diff_PPV_list_with_after_psm_values[i] / np.sqrt(num_of_bts))
        for i in range(len(Mean_Diff_PPV_list_with_after_psm_values))]
    p_value_PPV_diffFromMean_level_wise_psm = [scipy.stats.t.sf(np.abs(x), num_of_bts - 1) for x in
                                               t_stat_PPV_diffFromMean_level_wise_psm]


    Sensitivity_Sens_group_averages = np.zeros((num_of_bts, 3))
    Sensitivity_Diff_from_average = np.zeros((num_of_bts, 3, 3))
    for i in range(num_of_bts):
        for j in range(3):  # 3 because there are 3 sensitive attributes
            Sensitivity_Sens_group_averages[i, j] = np.sum(Sensitivity_level_wise[i, :, j]) /np.count_nonzero(Sensitivity_level_wise[i, :, j])
            for k in range(3):
                if Sensitivity_level_wise[i, k, j] != 0:
                    Sensitivity_Diff_from_average[i,k,j] = Sensitivity_level_wise[i, k, j] - Sensitivity_Sens_group_averages[i, j]

    Sensitivity_Sens_group_averages_across_bts = np.mean(Sensitivity_Sens_group_averages, axis=0)

    Mean_Diff_Sensitivity_list = []
    STD_Diff_Sensitivity_list = []
    ctr_sv = 0
    for sv in sens_var:
        ctr_l = 0
        for name in sens_var_levels:
            if name.split("_")[0] == sv:
                Mean_Diff_Sensitivity_list.append(
                    np.round(np.mean(Sensitivity_Diff_from_average[:, ctr_l, ctr_sv]), decimals=2))
                # Mean_Sensitivity_list_with_after_psm_values.append(
                #     np.round(np.mean(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
                STD_Diff_Sensitivity_list.append(
                    np.round(np.std(Sensitivity_Diff_from_average[:, ctr_l, ctr_sv]), decimals=4))
                # STD_Sensitivity_list_with_after_psm_values.append(
                #     np.round(np.std(Sensitivity_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
                ctr_l = ctr_l + 1
        ctr_sv = ctr_sv + 1

    # computing the t statistic to test significance in the difference of mean from average (single statistic)

    t_stat_Sensitivity_diffFromMean_level_wise = [Mean_Diff_Sensitivity_list[i]/(STD_Diff_Sensitivity_list[i]/np.sqrt(num_of_bts)) for i in range(len(Mean_Diff_Sensitivity_list))]
    p_value_Sensitivity_diffFromMean_level_wise = [scipy.stats.t.sf(np.abs(x), num_of_bts - 1) for x in t_stat_Sensitivity_diffFromMean_level_wise]


    Sensitivity_Sens_group_averages_psm = np.zeros((num_of_bts, 6))
    Sensitivity_Diff_from_average_psm = np.zeros((num_of_bts, 3, 6))
    for i in range(num_of_bts):
        for j in range(6):  # 6 because there are 6 psm groups
            Sensitivity_Sens_group_averages_psm[i, j] = np.sum(
                Sensitivity_level_wise_after_psm[i, :, j]) / 2  # because in psm we are matching 2 levels at a time
            for k in range(3):
                if Sensitivity_level_wise_after_psm[i,k,j] !=0:
                    Sensitivity_Diff_from_average_psm[i, k, j] = Sensitivity_level_wise_after_psm[i, k, j] - \
                                                             Sensitivity_Sens_group_averages_psm[i, j]

    Sensitivity_Sens_group_averages_across_bts_psm = np.mean(Sensitivity_Sens_group_averages_psm, axis=0)

    Sensitivity_Diff_from_average_across_bts_level_wise_psm = np.mean(Sensitivity_Diff_from_average_psm, axis=0)
    Sensitivity_Diff_from_std_across_bts_level_wise_psm = np.std(Sensitivity_Diff_from_average_psm, axis=0)

    Mean_Diff_Sensitivity_list_with_after_psm_values = [np.round(Sensitivity_Diff_from_average_across_bts_level_wise_psm[i,j], decimals=6) for j in range(6) for i in range(3)]
    STD_Diff_Sensitivity_list_with_after_psm_values = [np.round(Sensitivity_Diff_from_std_across_bts_level_wise_psm[i,j], decimals=4) for j in range(6) for i in range(3)]


    Mean_Diff_Sensitivity_list_with_after_psm_values = list(
        filter(lambda num: num != 0, Mean_Diff_Sensitivity_list_with_after_psm_values))
    STD_Diff_Sensitivity_list_with_after_psm_values = list(
        filter(lambda num: num != 0, STD_Diff_Sensitivity_list_with_after_psm_values))

    t_stat_Sensitivity_diffFromMean_level_wise_psm = [Mean_Diff_Sensitivity_list_with_after_psm_values[i]/(STD_Diff_Sensitivity_list_with_after_psm_values[i]/np.sqrt(num_of_bts)) for i in range(len(Mean_Diff_Sensitivity_list_with_after_psm_values))]
    p_value_Sensitivity_diffFromMean_level_wise_psm = [scipy.stats.t.sf(np.abs(x), num_of_bts - 1) for x in t_stat_Sensitivity_diffFromMean_level_wise_psm]


    AUROC_Sens_group_averages = np.zeros((num_of_bts, 3))
    AUROC_Diff_from_average = np.zeros((num_of_bts, 3, 3))
    for i in range(num_of_bts):
        for j in range(3):  # 3 because there are 3 sensitive attributes
            AUROC_Sens_group_averages[i, j] = np.sum(AUROC_level_wise[i, :, j]) / np.count_nonzero(AUROC_level_wise[i, :, j])
            for k in range(3):
                if AUROC_level_wise[i, k, j] != 0:
                    AUROC_Diff_from_average[i, k, j] = AUROC_level_wise[i, k, j] - AUROC_Sens_group_averages[i, j]
    AUROC_Sens_group_averages_across_bts = np.mean(AUROC_Sens_group_averages, axis=0)

    Mean_Diff_AUROC_list = []
    STD_Diff_AUROC_list = []
    ctr_sv = 0
    for sv in sens_var:
        ctr_l = 0
        for name in sens_var_levels:
            if name.split("_")[0] == sv:
                print(sv, np.mean(AUROC_Diff_from_average[:, ctr_l, ctr_sv]))
                Mean_Diff_AUROC_list.append(
                    np.round(np.mean(AUROC_Diff_from_average[:, ctr_l, ctr_sv]), decimals=2))
                # Mean_AUROC_list_with_after_psm_values.append(
                #     np.round(np.mean(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
                STD_Diff_AUROC_list.append(
                    np.round(np.std(AUROC_Diff_from_average[:, ctr_l, ctr_sv]), decimals=4))
                # STD_AUROC_list_with_after_psm_values.append(
                #     np.round(np.std(AUROC_level_wise_after_psm[:, ctr_l, ctr_sv]), decimals=4))
                ctr_l = ctr_l + 1
        ctr_sv = ctr_sv + 1
    # computing the t statistic to test significance in the difference of mean fomr average (single statistic)
    t_stat_AUROC_diffFromMean_level_wise = [Mean_Diff_AUROC_list[i] / (STD_Diff_AUROC_list[i] / np.sqrt(num_of_bts)) for
                                            i in
                                            range(len(Mean_Diff_AUROC_list))]
    p_value_AUROC_diffFromMean_level_wise = [scipy.stats.t.sf(np.abs(x), num_of_bts - 1) for x in
                                             t_stat_AUROC_diffFromMean_level_wise]

    AUROC_Sens_group_averages_psm = np.zeros((num_of_bts, 6))
    AUROC_Diff_from_average_psm = np.zeros((num_of_bts, 3, 6))
    for i in range(num_of_bts):
        for j in range(6):  # 6 because there are 6 psm groups
            AUROC_Sens_group_averages_psm[i, j] = np.sum(
                AUROC_level_wise_after_psm[i, :, j]) / 2  # because in psm we are matching 2 levels at a time
            for k in range(3):
                if AUROC_level_wise_after_psm[i, k, j] != 0:
                    AUROC_Diff_from_average_psm[i, k, j] = AUROC_level_wise_after_psm[i, k, j] - \
                                                           AUROC_Sens_group_averages_psm[i, j]
    AUROC_Sens_group_averages_across_bts_psm = np.mean(AUROC_Sens_group_averages_psm, axis=0)

    AUROC_Diff_from_average_across_bts_level_wise_psm = np.mean(AUROC_Diff_from_average_psm, axis=0)
    AUROC_Diff_from_std_across_bts_level_wise_psm = np.std(AUROC_Diff_from_average_psm, axis=0)

    Mean_Diff_AUROC_list_with_after_psm_values = [
        np.round(AUROC_Diff_from_average_across_bts_level_wise_psm[i, j], decimals=6) for j in range(6) for i in
        range(3)]
    STD_Diff_AUROC_list_with_after_psm_values = [
        np.round(AUROC_Diff_from_std_across_bts_level_wise_psm[i, j], decimals=4) for j in range(6) for i in range(3)]

    Mean_Diff_AUROC_list_with_after_psm_values = list(
        filter(lambda num: num != 0, Mean_Diff_AUROC_list_with_after_psm_values))
    STD_Diff_AUROC_list_with_after_psm_values = list(
        filter(lambda num: num != 0, STD_Diff_AUROC_list_with_after_psm_values))
    t_stat_AUROC_diffFromMean_level_wise_psm = [
        Mean_Diff_AUROC_list_with_after_psm_values[i] / (
                    STD_Diff_AUROC_list_with_after_psm_values[i] / np.sqrt(num_of_bts))
        for i in range(len(Mean_Diff_AUROC_list_with_after_psm_values))]
    p_value_AUROC_diffFromMean_level_wise_psm = [scipy.stats.t.sf(np.abs(x), num_of_bts - 1) for x in
                                                 t_stat_AUROC_diffFromMean_level_wise_psm]


    return true_outcome_rate, frequency_level_wise, frequency_level_wise_psm, freq_level_del, sens_var_levels, sens_var_level_dict, \
           Mean_Diff_PPV_list, Mean_Diff_PPV_list_with_after_psm_values, p_value_PPV_diffFromMean_level_wise, p_value_PPV_diffFromMean_level_wise_psm, \
           Mean_Diff_Sensitivity_list, Mean_Diff_Sensitivity_list_with_after_psm_values, p_value_Sensitivity_diffFromMean_level_wise, p_value_Sensitivity_diffFromMean_level_wise_psm, \
           Mean_Diff_AUROC_list, Mean_Diff_AUROC_list_with_after_psm_values, p_value_AUROC_diffFromMean_level_wise, p_value_AUROC_diffFromMean_level_wise_psm, \
           PPV_Sens_group_averages_across_bts, PPV_Sens_group_averages_across_bts_psm, Sensitivity_Sens_group_averages_across_bts, Sensitivity_Sens_group_averages_across_bts_psm, \
           AUROC_Sens_group_averages_across_bts, AUROC_Sens_group_averages_across_bts_psm, np.mean(
        Mean_PPV_list), np.mean(Mean_Sensitivity_list), np.mean(Mean_AUROC_list)


def Generating_pdf_simple(true_outcome_rate, frequency_level_wise,
                   frequency_level_wise_psm, freq_level_del):
    """  **************************************************************************************** """
    """  Getting the final pdf  """
    """  **************************************************************************************** """
    #
    # model_details_list = []
    # use_list = []
    # factors_list = []
    # metrics_list = []
    # caveatlist = []
    # cohort_details_list = []
    # calibration_list = []
    # utility_des_list =[]
    #
    # # print(len(txtfile_From_dev))
    # for i in range(len(txtfile_From_dev)):
    #     # print(txtfile_From_dev[i].split()[0], i)
    #     if txtfile_From_dev[i] == "## Model details\n":
    #         model_details_list.append(txtfile_From_dev[i + 1])
    #     if txtfile_From_dev[i] == "## Intended use\n":
    #         use_list.append(txtfile_From_dev[i + 1])
    #     if txtfile_From_dev[i] == "## Factors\n":
    #         factors_list.append(txtfile_From_dev[i + 1])
    #     if txtfile_From_dev[i] == "## Metrics\n":
    #         metrics_list.append(txtfile_From_dev[i + 1])
    #     if txtfile_From_dev[i] == "## Cohort details\n":
    #         cohort_details_list.append(txtfile_From_dev[i + 1])
    #     if txtfile_From_dev[i] == "## Caveats\n":
    #         caveatlist.append(txtfile_From_dev[i + 1])
    #     if txtfile_From_dev[i] == "## Calibration\n":
    #         calibration_list.append(txtfile_From_dev[i + 1])
    #     if txtfile_From_dev[i] == "## Utility calculation\n":
    #         utility_des_list.append(txtfile_From_dev[i + 1])
    #
    # # making the quantity based utility to qualitative terms
    # # utility_intervals = pd.qcut(utility_dif_c1, 3, labels=['small', 'medium', 'high'])
    # # utility_intervals_list = list(pd.qcut(utility_dif_c1, 3).unique().sort_values())
    # # interP = ['small', 'medium', 'high']
    # # dict_Quant_to_Qual = dict(zip(utility_intervals_list, interP))
    #
    # neg_exp_utility_list = []
    # pos_exp_utility_list = []
    # for i in expected_node_value:
    #     if i < 0:
    #         neg_exp_utility_list.append(i)
    #     else:
    #         pos_exp_utility_list.append(np.round(i, decimals=3))
    #
    # # converting the quantity based expected utility to qualitative terms
    # utility_intervals_value = pd.qcut(pos_exp_utility_list, 3, labels=['low', 'moderate', 'high'])
    # intervals_for_utility = list(pd.qcut(pos_exp_utility_list, 3).unique().sort_values())
    # utility_intervals_list_val = []
    # for i in range(len(intervals_for_utility)):
    #     left_int = np.round(intervals_for_utility[i].left, decimals = 3)
    #     right_int = np.round(intervals_for_utility[i].right, decimals = 3)
    #     utility_intervals_list_val.append("("+str(left_int)+", "+str(right_int)+"]")
    # interP_val = ['low', 'moderate', 'high']
    # dict_Quant_to_Qual_val = dict(zip(utility_intervals_list_val, interP_val))
    #
    # if expected_node_value_individual_patient < 0:
    #     patient_group = 'Negative'
    # else:
    #     patient_group = utility_intervals_value[pos_exp_utility_list.index(np.round(expected_node_value_individual_patient, decimals=3))]


    temp_feature_list_psm = [i for i in frequency_level_wise_psm.keys() if frequency_level_wise_psm[i] >100]
    # temp_feature_list_psm.remove("RACE_0psm_")

    for i in sens_var_level_dict.keys():
        if sens_var_level_dict[i] == 'S_Others/NA':
            sens_var_level_dict[i] = 'Sex Information NA'
        if sens_var_level_dict[i] == 'R_Others/NA':
            sens_var_level_dict[i] = 'Race Information NA'
        if sens_var_level_dict[i].split("[")[0] == 'Age\n ':
            sens_var_level_dict[i] = 'Age group [' + sens_var_level_dict[i].split("[")[1]

    # getting the Table form for PPV
    data_PPV = pd.DataFrame(dtype=object)
    data_PPV['Feature_level'] = sens_var_level
    data_PPV['PPV_diff_from_avg'] = Avg_PPV_Diff
    data_PPV['PPV_diff_from_avg'] = data_PPV['PPV_diff_from_avg'].astype(object)
    data_PPV['PPV_diff_from_avg_psm'] = 0
    data_PPV['PPV_diff_from_avg_psm'] = data_PPV['PPV_diff_from_avg_psm'].astype(object)
    data_PPV.set_index('Feature_level', inplace=True)
    Race_1_psm_list_PPV = []
    Race_0_psm_list_PPV = []
    Race_nan_psm_list_PPV = []
    age_0_psm_list_PPV = []
    age_1_psm_list_PPV = []
    age_2_psm_list_PPV = []

    temp_PPV_avg_diff_list = dict(zip(temp_feature_list_psm, Avg_PPV_Diff_psm))
    temp_PPV_p_value_diff_list = dict(zip(temp_feature_list_psm, p_PPV_Diff_psm))
    for sv in sens_var:
        for name in temp_PPV_avg_diff_list.keys():
            if (sv == 'Sex') & (name.split("_")[0] == sv):
                if temp_PPV_p_value_diff_list[name] < 0.05:
                    data_PPV.at[name, 'PPV_diff_from_avg_psm'] = str( np.round(temp_PPV_avg_diff_list[name], decimals=2)) + "*"
                else:
                    data_PPV.at[name, 'PPV_diff_from_avg_psm'] =  np.round(temp_PPV_avg_diff_list[name], decimals=2)
            if sv == 'RACE':
                counter_race_7 = 0
                if (name.split("_")[1] == '1psm') & (counter_race_7 == 0):
                    if temp_PPV_avg_diff_list[name] != 0:
                        if temp_PPV_p_value_diff_list[name] < 0.05:
                            Race_1_psm_list_PPV.append(str( np.round(temp_PPV_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            Race_1_psm_list_PPV.append( np.round(temp_PPV_avg_diff_list[name], decimals=2))
                    data_PPV.at['RACE_1', 'PPV_diff_from_avg_psm'] = Race_1_psm_list_PPV
                    counter_race_7 = 1
                counter_race_9 = 0
                if (name.split("_")[1] == '0psm') & (counter_race_9 == 0):
                    if temp_PPV_avg_diff_list[name] != 0:
                        if temp_PPV_p_value_diff_list[name] < 0.05:
                            Race_0_psm_list_PPV.append(str( np.round(temp_PPV_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            Race_0_psm_list_PPV.append( np.round(temp_PPV_avg_diff_list[name], decimals=2))
                    data_PPV.at['RACE_0', 'PPV_diff_from_avg_psm'] = Race_0_psm_list_PPV
                    counter_race_9 = 1
                counter_race_nan = 0
                if (name.split("_")[1] == '-1psm') & (counter_race_nan == 0):
                    if temp_PPV_avg_diff_list[name] != 0:
                        if temp_PPV_p_value_diff_list[name] < 0.05:
                            Race_nan_psm_list_PPV.append(str( np.round(temp_PPV_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            Race_nan_psm_list_PPV.append( np.round(temp_PPV_avg_diff_list[name], decimals=2))
                    data_PPV.at['RACE_-1', 'PPV_diff_from_avg_psm'] = Race_nan_psm_list_PPV
                    counter_race_nan = 1
            if sv == 'age':
                counter_age_0 = 0
                if (name.split(".")[0] == 'age_0') & (counter_age_0 == 0):
                    if temp_PPV_avg_diff_list[name] != 0:
                        if temp_PPV_p_value_diff_list[name] < 0.05:
                            age_0_psm_list_PPV.append(str( np.round(temp_PPV_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            age_0_psm_list_PPV.append( np.round(temp_PPV_avg_diff_list[name], decimals=2))
                        data_PPV.at['age_0.0', 'PPV_diff_from_avg_psm'] = age_0_psm_list_PPV[0]
                    counter_age_0 = 1
                counter_age_1 = 0
                if (name.split(".")[0] == 'age_1') & (counter_age_1 == 0):
                    if temp_PPV_avg_diff_list[name] != 0:
                        if temp_PPV_p_value_diff_list[name] < 0.05:
                            age_1_psm_list_PPV.append(str( np.round(temp_PPV_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            age_1_psm_list_PPV.append( np.round(temp_PPV_avg_diff_list[name], decimals=2))
                    data_PPV.at['age_1.0', 'PPV_diff_from_avg_psm'] = age_1_psm_list_PPV
                    counter_age_1 = 1
                counter_age_2 = 0
                if (name.split(".")[0] == 'age_2') & (counter_age_2 == 0):
                    if temp_PPV_avg_diff_list[name] != 0:
                        if temp_PPV_p_value_diff_list[name] < 0.05:
                            age_2_psm_list_PPV.append(str( np.round(temp_PPV_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            age_2_psm_list_PPV.append( np.round(temp_PPV_avg_diff_list[name], decimals=2))
                        data_PPV.at['age_2.0', 'PPV_diff_from_avg_psm'] = age_2_psm_list_PPV[0]
                    counter_age_2 = 1


    data_PPV['Feature_level_inter'] = [sens_var_level_dict[i] for i in sens_var_level_dict.keys()]
    data_PPV = data_PPV[['Feature_level_inter', 'PPV_diff_from_avg', 'PPV_diff_from_avg_psm']]

    # getting the Table form for Sensitivity

    data_Sensitivity = pd.DataFrame(dtype=object)
    data_Sensitivity['Feature_level'] = sens_var_level
    data_Sensitivity['Sensitivity_diff_from_avg'] = Avg_Sensitivity_Diff
    data_Sensitivity['Sensitivity_diff_from_avg'] = data_Sensitivity['Sensitivity_diff_from_avg'].astype(object)
    data_Sensitivity['Sensitivity_diff_from_avg_psm'] = 0
    data_Sensitivity['Sensitivity_diff_from_avg_psm'] = data_Sensitivity['Sensitivity_diff_from_avg_psm'].astype(object)
    data_Sensitivity.set_index('Feature_level', inplace=True)
    Race_1_psm_list_Sensitivity = []
    Race_0_psm_list_Sensitivity = []
    Race_nan_psm_list_Sensitivity = []
    age_0_psm_list_Sensitivity = []
    age_1_psm_list_Sensitivity = []
    age_2_psm_list_Sensitivity = []

    temp_Sensitivity_avg_diff_list = dict(zip(temp_feature_list_psm, Avg_Sensitivity_Diff_psm))
    temp_Sensitivity_p_value_diff_list = dict(zip(temp_feature_list_psm, p_Sensitivity_Diff_psm))
    for sv in sens_var:
        for name in temp_Sensitivity_avg_diff_list.keys():
            if (sv == 'Sex') & (name.split("_")[0] == sv):
                if temp_Sensitivity_p_value_diff_list[name] < 0.05:
                    data_Sensitivity.at[name, 'Sensitivity_diff_from_avg_psm'] = str(
                         np.round(temp_Sensitivity_avg_diff_list[name], decimals=2)) + "*"
                else:
                    data_Sensitivity.at[name, 'Sensitivity_diff_from_avg_psm'] =  np.round(temp_Sensitivity_avg_diff_list[name], decimals=2)
            if sv == 'RACE':
                counter_race_7 = 0
                if (name.split("_")[1] == '1psm') & (counter_race_7 == 0):
                    if temp_Sensitivity_avg_diff_list[name] != 0:
                        if temp_Sensitivity_p_value_diff_list[name] < 0.05:
                            Race_1_psm_list_Sensitivity.append(str( np.round(temp_Sensitivity_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            Race_1_psm_list_Sensitivity.append( np.round(temp_Sensitivity_avg_diff_list[name], decimals=2))
                    data_Sensitivity.at['RACE_1', 'Sensitivity_diff_from_avg_psm'] = Race_1_psm_list_Sensitivity
                    counter_race_7 = 1
                counter_race_9 = 0
                if (name.split("_")[1] == '0psm') & (counter_race_9 == 0):
                    if temp_Sensitivity_avg_diff_list[name] != 0:
                        if temp_Sensitivity_p_value_diff_list[name] < 0.05:
                            Race_0_psm_list_Sensitivity.append(str( np.round(temp_Sensitivity_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            Race_0_psm_list_Sensitivity.append( np.round(temp_Sensitivity_avg_diff_list[name], decimals=2))
                    data_Sensitivity.at['RACE_0', 'Sensitivity_diff_from_avg_psm'] = Race_0_psm_list_Sensitivity
                    counter_race_9 = 1
                counter_race_nan = 0
                if (name.split("_")[1] == '-1psm') & (counter_race_nan == 0):
                    if temp_Sensitivity_avg_diff_list[name] != 0:
                        if temp_Sensitivity_p_value_diff_list[name] < 0.05:
                            Race_nan_psm_list_Sensitivity.append(str( np.round(temp_Sensitivity_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            Race_nan_psm_list_Sensitivity.append( np.round(temp_Sensitivity_avg_diff_list[name], decimals=2))
                    data_Sensitivity.at['RACE_-1', 'Sensitivity_diff_from_avg_psm'] = Race_nan_psm_list_Sensitivity
                    counter_race_nan = 1
            if sv == 'age':
                counter_age_0 = 0
                if (name.split(".")[0] == 'age_0') & (counter_age_0 == 0):
                    if temp_Sensitivity_avg_diff_list[name] != 0:
                        if temp_Sensitivity_p_value_diff_list[name] < 0.05:
                            age_0_psm_list_Sensitivity.append(str( np.round(temp_Sensitivity_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            age_0_psm_list_Sensitivity.append( np.round(temp_Sensitivity_avg_diff_list[name], decimals=2))
                        data_Sensitivity.at['age_0.0', 'Sensitivity_diff_from_avg_psm'] = age_0_psm_list_Sensitivity[0]
                    counter_age_0 = 1
                counter_age_1 = 0
                if (name.split(".")[0] == 'age_1') & (counter_age_1 == 0):
                    if temp_Sensitivity_avg_diff_list[name] != 0:
                        if temp_Sensitivity_p_value_diff_list[name] < 0.05:
                            age_1_psm_list_Sensitivity.append(str( np.round(temp_Sensitivity_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            age_1_psm_list_Sensitivity.append( np.round(temp_Sensitivity_avg_diff_list[name], decimals=2))
                    data_Sensitivity.at['age_1.0', 'Sensitivity_diff_from_avg_psm'] = age_1_psm_list_Sensitivity
                    counter_age_1 = 1
                counter_age_2 = 0
                if (name.split(".")[0] == 'age_2') & (counter_age_2 == 0):
                    if temp_Sensitivity_avg_diff_list[name] != 0:
                        if temp_Sensitivity_p_value_diff_list[name] < 0.05:
                            age_2_psm_list_Sensitivity.append(str( np.round(temp_Sensitivity_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            age_2_psm_list_Sensitivity.append( np.round(temp_Sensitivity_avg_diff_list[name], decimals=2))
                        data_Sensitivity.at['age_2.0', 'Sensitivity_diff_from_avg_psm'] = age_2_psm_list_Sensitivity[0]
                    counter_age_2 = 1


    data_Sensitivity = data_Sensitivity[['Sensitivity_diff_from_avg', 'Sensitivity_diff_from_avg_psm']]

    data_PPV = data_PPV.join(data_Sensitivity)

    # getting the Table form for AUROC

    data_AUROC = pd.DataFrame(dtype=object)
    data_AUROC['Feature_level'] = sens_var_level
    data_AUROC['AUROC_diff_from_avg'] = Avg_AUROC_Diff
    data_AUROC['AUROC_diff_from_avg'] = data_AUROC['AUROC_diff_from_avg'].astype(object)
    data_AUROC['AUROC_diff_from_avg_psm'] = 0
    data_AUROC['AUROC_diff_from_avg_psm'] = data_AUROC['AUROC_diff_from_avg_psm'].astype(object)
    data_AUROC['AUROC_diff_from_avg_psmOnlyval'] = 0
    data_AUROC['AUROC_diff_from_avg_psmOnlyval'] = data_AUROC['AUROC_diff_from_avg_psmOnlyval'].astype(object)
    data_AUROC['feature'] = 0
    data_AUROC['feature'] = data_AUROC['feature'].astype(object)
    data_AUROC.set_index('Feature_level', inplace=True)
    Race_1_psm_list_AUROC = []
    Race_0_psm_list_AUROC = []
    Race_nan_psm_list_AUROC = []
    age_0_psm_list_AUROC = []
    age_1_psm_list_AUROC = []
    age_2_psm_list_AUROC = []

    temp_AUROC_avg_diff_list = dict(zip(temp_feature_list_psm, Avg_AUROC_Diff_psm))
    temp_AUROC_p_value_diff_list = dict(zip(temp_feature_list_psm, p_AUROC_Diff_psm))
    for sv in sens_var:
        for name in temp_AUROC_avg_diff_list.keys():
            if (sv == 'Sex') & (name.split("_")[0] == sv):
                if temp_AUROC_p_value_diff_list[name] < 0.05:
                    data_AUROC.at[name, 'AUROC_diff_from_avg_psm'] = str(
                         np.round(temp_AUROC_avg_diff_list[name], decimals=2)) + "*"
                else:
                    data_AUROC.at[name, 'AUROC_diff_from_avg_psm'] =  np.round(temp_AUROC_avg_diff_list[name], decimals=2)
                data_AUROC.at[name, 'feature'] = sv
                data_AUROC.at[name, 'AUROC_diff_from_avg_psmOnlyval'] = np.round(temp_AUROC_avg_diff_list[name], decimals=2)


            if sv == 'RACE':
                counter_race_7 = 0
                if (name.split("_")[1] == '1psm') & (counter_race_7 == 0):
                    if temp_AUROC_avg_diff_list[name] != 0:
                        if temp_AUROC_p_value_diff_list[name] < 0.05:
                            Race_1_psm_list_AUROC.append(str( np.round(temp_AUROC_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            Race_1_psm_list_AUROC.append( np.round(temp_AUROC_avg_diff_list[name], decimals=2))
                    data_AUROC.at['RACE_1', 'AUROC_diff_from_avg_psm'] = Race_1_psm_list_AUROC
                    data_AUROC.at['RACE_1', 'feature'] = sv
                    data_AUROC.at['RACE_1', 'AUROC_diff_from_avg_psmOnlyval'] = str(data_AUROC.at['RACE_1', 'AUROC_diff_from_avg_psmOnlyval']) + "," +str( np.round(temp_AUROC_avg_diff_list[name], decimals=2))


                    counter_race_7 = 1
                counter_race_9 = 0
                if (name.split("_")[1] == '0psm') & (counter_race_9 == 0):
                    if temp_AUROC_avg_diff_list[name] != 0:
                        if temp_AUROC_p_value_diff_list[name] < 0.05:
                            Race_0_psm_list_AUROC.append(str( np.round(temp_AUROC_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            Race_0_psm_list_AUROC.append( np.round(temp_AUROC_avg_diff_list[name], decimals=2))
                    data_AUROC.at['RACE_0', 'AUROC_diff_from_avg_psm'] = Race_0_psm_list_AUROC
                    data_AUROC.at['RACE_0', 'feature'] = sv
                    data_AUROC.at['RACE_0', 'AUROC_diff_from_avg_psmOnlyval'] = str(data_AUROC.at['RACE_0', 'AUROC_diff_from_avg_psmOnlyval']) + "," +str( np.round(temp_AUROC_avg_diff_list[name], decimals=2))


                    counter_race_9 = 1
                counter_race_nan = 0
                if (name.split("_")[1] == '-1psm') & (counter_race_nan == 0):
                    if temp_AUROC_avg_diff_list[name] != 0:
                        if temp_AUROC_p_value_diff_list[name] < 0.05:
                            Race_nan_psm_list_AUROC.append(str( np.round(temp_AUROC_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            Race_nan_psm_list_AUROC.append( np.round(temp_AUROC_avg_diff_list[name], decimals=2))
                    data_AUROC.at['RACE_-1', 'AUROC_diff_from_avg_psm'] = Race_nan_psm_list_AUROC
                    data_AUROC.at['RACE_-1', 'feature'] = sv
                    data_AUROC.at['RACE_-1', 'AUROC_diff_from_avg_psmOnlyval'] = str(data_AUROC.at['RACE_-1', 'AUROC_diff_from_avg_psmOnlyval']) + "," +str( np.round(temp_AUROC_avg_diff_list[name], decimals=2))


                    counter_race_nan = 1
            if sv == 'age':
                counter_age_0 = 0
                if (name.split(".")[0] == 'age_0') & (counter_age_0 == 0):
                    if temp_AUROC_avg_diff_list[name] != 0:
                        if temp_AUROC_p_value_diff_list[name] < 0.05:
                            age_0_psm_list_AUROC.append(str( np.round(temp_AUROC_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            age_0_psm_list_AUROC.append( np.round(temp_AUROC_avg_diff_list[name], decimals=2))
                        data_AUROC.at['age_0.0', 'AUROC_diff_from_avg_psm'] = age_0_psm_list_AUROC[0]
                        data_AUROC.at['age_0.0', 'feature'] = sv
                        data_AUROC.at['age_0.0', 'AUROC_diff_from_avg_psmOnlyval'] = str(
                            data_AUROC.at['age_0.0', 'AUROC_diff_from_avg_psmOnlyval']) + "," + str(
                            np.round(temp_AUROC_avg_diff_list[name], decimals=2))

                    counter_age_0 = 1
                counter_age_1 = 0
                if (name.split(".")[0] == 'age_1') & (counter_age_1 == 0):
                    if temp_AUROC_avg_diff_list[name] != 0:
                        if temp_AUROC_p_value_diff_list[name] < 0.05:
                            age_1_psm_list_AUROC.append(str( np.round(temp_AUROC_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            age_1_psm_list_AUROC.append( np.round(temp_AUROC_avg_diff_list[name], decimals=2))
                    data_AUROC.at['age_1.0', 'AUROC_diff_from_avg_psm'] = age_1_psm_list_AUROC
                    data_AUROC.at['age_1.0', 'feature'] = sv
                    data_AUROC.at['age_1.0', 'AUROC_diff_from_avg_psmOnlyval'] = str(
                        data_AUROC.at['age_1.0', 'AUROC_diff_from_avg_psmOnlyval']) + "," + str(
                        np.round(temp_AUROC_avg_diff_list[name], decimals=2))

                    counter_age_1 = 1
                counter_age_2 = 0
                if (name.split(".")[0] == 'age_2') & (counter_age_2 == 0):
                    if temp_AUROC_avg_diff_list[name] != 0:
                        if temp_AUROC_p_value_diff_list[name] < 0.05:
                            age_2_psm_list_AUROC.append(str( np.round(temp_AUROC_avg_diff_list[name], decimals=2)) + "*")
                        else:
                            age_2_psm_list_AUROC.append( np.round(temp_AUROC_avg_diff_list[name], decimals=2))
                        data_AUROC.at['age_2.0', 'AUROC_diff_from_avg_psm'] = age_2_psm_list_AUROC[0]
                        data_AUROC.at['age_2.0', 'feature'] = sv
                        data_AUROC.at['age_2.0', 'AUROC_diff_from_avg_psmOnlyval'] = str(
                            data_AUROC.at['age_2.0', 'AUROC_diff_from_avg_psmOnlyval']) + "," + str(
                            np.round(temp_AUROC_avg_diff_list[name], decimals=2))

                    counter_age_2 = 1

    data_sum_AUROC = data_AUROC[['AUROC_diff_from_avg', 'feature','AUROC_diff_from_avg_psmOnlyval']]
    summary_Table = data_sum_AUROC.groupby('feature').agg(
        {'AUROC_diff_from_avg': ['max', 'min'], 'AUROC_diff_from_avg_psmOnlyval': ['max', 'min']})
    summary_Table['gap'] = summary_Table[('AUROC_diff_from_avg', 'max')] - summary_Table[('AUROC_diff_from_avg', 'min')]
    summary_Table[('AUROC_diff_from_avg_psmOnlyval', 'max')] = [float(max(str(a).split(","))) for a in summary_Table[('AUROC_diff_from_avg_psmOnlyval', 'max')]]
    summary_Table[('AUROC_diff_from_avg_psmOnlyval', 'min')] = [float(min(str(a).split(","))) for a in summary_Table[('AUROC_diff_from_avg_psmOnlyval', 'min')]]
    summary_Table['gap_psm'] = summary_Table[('AUROC_diff_from_avg_psmOnlyval', 'max')] - summary_Table[('AUROC_diff_from_avg_psmOnlyval', 'min')]

    summary_Table = summary_Table[['gap', 'gap_psm']]
    summary_Table.to_csv('./'+ data_version +'_summary_auroc' +str(num_of_bts) +'.csv', index=True)

    data_AUROC = data_AUROC[['AUROC_diff_from_avg', 'AUROC_diff_from_avg_psm']]

    data_PPV = data_PPV.join(data_AUROC)

    idx_ctr = 0
    for index in data_PPV.index:
        if p_PPV_Diff[idx_ctr] < 0.05:
            data_PPV.at[index, 'PPV_diff_from_avg'] = str(Avg_PPV_Diff[idx_ctr]) + "*"
        if p_Sensitivity_Diff[idx_ctr]< 0.05:
            data_PPV.at[index, 'Sensitivity_diff_from_avg'] = str(Avg_Sensitivity_Diff[idx_ctr]) + "*"
        if p_AUROC_Diff[idx_ctr]< 0.05:
            data_PPV.at[index, 'AUROC_diff_from_avg'] = str(Avg_AUROC_Diff[idx_ctr]) + "*"
        idx_ctr = idx_ctr+1

    Grp_avg_PPV_psm1 = [np.round(Grp_avg_PPV_psm[i], decimals=2) for i in range(len(Grp_avg_PPV_psm))]
    Grp_avg_Sensitivity_psm1 = [np.round(Grp_avg_Sensitivity_psm[i], decimals=2) for i in range(len(Grp_avg_Sensitivity_psm))]

    # for ease of making a table in fpdf
    data_PPV = data_PPV.append(dict(zip(data_PPV.columns,
                                        ['Patient group', 'Grp PPV - Avg PPV', 'Grp PPV - Avg PPV',
                                         'Grp Sensitivity - Avg Sensitivity', 'Grp Sensitivity - Avg Sensitivity',
                                         'Grp AUROC - Avg AUROC', 'Grp AUROC - Avg AUROC'])),
                               ignore_index=True)
    data_PPV = data_PPV.append(dict(zip(data_PPV.columns,
                                        [' ', ' ', '(After matching ^^)',
                                         ' ', '(After matching ^^)',
                                         ' ', '(After matching ^^)'])),
                               ignore_index=True)
    data_PPV = data_PPV.append(dict(zip(data_PPV.columns,
                                        ['Overall Avg', str(np.round(overall_PPV, decimals=2)) + " (PPV)", "-",
                                         str(np.round(overall_Sensitivity, decimals=2)) + " (Sensitivity)", "-",
                                         str(np.round(overall_AUROC, decimals=2)) + " (AUROC)", "-"])),
                               ignore_index=True)

    # data_PPV = data_PPV.append(dict(zip(data_PPV.columns,
    #                                     ['Sex (GrpAvg)', np.round(Grp_avg_PPV[0], decimals=2), Grp_avg_PPV_psm1[0],
    #                                      np.round(Grp_avg_Sensitivity[0], decimals=2), Grp_avg_Sensitivity_psm1[0]])),
    #                            ignore_index=True)
    # data_PPV = data_PPV.append(dict(zip(data_PPV.columns,
    #                                     ['Race (GrpAvg)', np.round(Grp_avg_PPV[1], decimals=2) , "["+ str(Grp_avg_PPV_psm1[1])+" , " + str(Grp_avg_PPV_psm1[2])+" , "+str(Grp_avg_PPV_psm1[3]) +"]",
    #                                      np.round(Grp_avg_Sensitivity[1], decimals=2), "["+ str(Grp_avg_Sensitivity_psm1[1])+" , " + str(Grp_avg_Sensitivity_psm1[2])+" , "+str(Grp_avg_Sensitivity_psm1[3]) +"]"])),
    #                            ignore_index=True)
    # data_PPV = data_PPV.append(dict(zip(data_PPV.columns,
    #                                     ['Age (GrpAvg)', np.round(Grp_avg_PPV[2], decimals=2), "["+ str(Grp_avg_PPV_psm1[4])+" , " + str(Grp_avg_PPV_psm1[5])+"]",
    #                                      np.round(Grp_avg_Sensitivity[2], decimals=2), "["+ str(Grp_avg_Sensitivity_psm1[4])+" , " + str(Grp_avg_Sensitivity_psm1[5])+"]"])),
    #                            ignore_index=True)

    data_PPV = data_PPV.reindex([ 8,9, 0, 1, 2, 3, 4, 5, 6, 7, 10])
    # data_PPV = data_PPV.reindex([8, 0, 1, 9, 2, 3, 4,10, 5, 6, 7,11])
    data_PPV_1 = data_PPV.values.tolist()

    # frequency table code

    data = pd.DataFrame(dtype=object)
    data['Feature_level'] = frequency_level_wise.keys()
    data['Patient_freq_basic'] = frequency_level_wise.values()
    data['Patient_freq_Del_pos'] = freq_level_del['Patient_freq_Delpos']
    data['Patient_freq_Del_neg'] = freq_level_del['Patient_freq_Delneg']
    data['Patient_freq_psm'] = 0
    data['Patient_freq_psm'] = data['Patient_freq_psm'].astype(object)
    data.set_index('Feature_level', inplace=True)
    Race_1_psm_list = []
    Race_0_psm_list = []
    Race_nan_psm_list = []
    age_0_psm_list = []
    age_1_psm_list = []
    age_2_psm_list = []

    for sv in sens_var:
        for name in frequency_level_wise_psm.keys():
            if (sv == 'Sex') & (name.split("_")[0] == sv):
                data.at[name, 'Patient_freq_psm'] = frequency_level_wise_psm[name]
            if sv == 'RACE':
                counter_race_7 = 0
                if (name.split("_")[1] == '1psm') & (counter_race_7 == 0):
                    if frequency_level_wise_psm[name] != 0:
                        Race_1_psm_list.append(frequency_level_wise_psm[name])
                    data.at['RACE_1', 'Patient_freq_psm'] = Race_1_psm_list
                    counter_race_7 = 1
                counter_race_9 = 0
                if (name.split("_")[1] == '0psm') & (counter_race_9 == 0):
                    if frequency_level_wise_psm[name] != 0:
                        Race_0_psm_list.append(frequency_level_wise_psm[name])
                    data.at['RACE_0', 'Patient_freq_psm'] = Race_0_psm_list
                    counter_race_9 = 1
                counter_race_nan = 0
                if (name.split("_")[1] == '-1psm') & (counter_race_nan == 0):
                    if frequency_level_wise_psm[name] != 0:
                        Race_nan_psm_list.append(frequency_level_wise_psm[name])
                    data.at['RACE_-1', 'Patient_freq_psm'] = Race_nan_psm_list
                    counter_race_nan = 1
            if sv == 'age':
                counter_age_0 = 0
                if (name.split(".")[0] == 'age_0') & (counter_age_0 == 0):
                    if frequency_level_wise_psm[name] != 0:
                        age_0_psm_list.append(frequency_level_wise_psm[name])
                        data.at['age_0.0', 'Patient_freq_psm'] = age_0_psm_list[0]
                    counter_age_0 = 1
                counter_age_1 = 0
                if (name.split(".")[0] == 'age_1') & (counter_age_1 == 0):
                    if frequency_level_wise_psm[name] != 0:
                        age_1_psm_list.append(frequency_level_wise_psm[name])
                    data.at['age_1.0', 'Patient_freq_psm'] = age_1_psm_list
                    counter_age_1 = 1
                counter_age_2 = 0
                if (name.split(".")[0] == 'age_2') & (counter_age_2 == 0):
                    if frequency_level_wise_psm[name] != 0:
                        age_2_psm_list.append(frequency_level_wise_psm[name])
                        data.at['age_2.0', 'Patient_freq_psm'] = age_2_psm_list[0]
                    counter_age_2 = 1

    data = data.loc[~(data == 0).all(axis=1)]  # dropping the race 8 category if it doesn't have any patients

    data = data.reindex(sens_var_level_dict.keys())
    data['Feature_level_inter'] = sens_var_level_dict.values()
    data = data[['Feature_level_inter', 'Patient_freq_basic','Patient_freq_Del_pos', 'Patient_freq_Del_neg', 'Patient_freq_psm']]
    # for ease of making a table in fpdf
    data = data.append(dict(zip(data.columns, ['Patient group', 'Group size total', 'Delirium group size', 'No Delirium group size' , 'Psm Group size total'])),
                       ignore_index=True)

    data = data.reindex([8, 0, 1, 2, 3, 4, 5, 6, 7])
    data_1 = data.values.tolist()

    data_PPV.to_csv('./'+ data_version +'_tabular_results_PPV_sens_auroc' +str(num_of_bts) +'.csv', index=False)
    data.to_csv('./'+ data_version +'_tabular_results_freq' +str(num_of_bts) +'.csv', index=False)

    exit()

    ##########################################################################################



random.seed(100)
task = 'postop_del'
model = 'RF'  # options from {LR, DT, GBT, RF, DNN}
model_dict = {'LR':'Logistic Regression', 'DT': 'Decision Trees', 'GBT': "Gradient Boosted Trees", 'RF': "Random Forests", 'DNN': "Deep Neural Network"}

data_version = 'MIMICwithsens' # options from {'MIMIC', 'MIMICwithsens', 'MIMICsub'}  # MIMICsub is the one that doesn't use sens variable

num_of_bts = 150

#### reading data files
x_validation_data = pd.read_csv("../Validation_data_for_cards/"+data_version+"_x_valid_" + str(model)+"_"+str(task)+ ".csv")
y_validation_data = pd.read_csv("../Validation_data_for_cards/"+data_version+"_y_true_valid_" + str(model)+"_"+str(task)+ ".csv")
y_validation_pred_full = pd.read_csv("../Validation_data_for_cards/"+data_version+"_y_pred_prob_valid_Full_" + str(model)+"_"+str(task)+ ".csv", header=None)
# y_validation_pred_basic = pd.read_csv("../Validation_data_for_cards/"+data_version+"_y_pred_prob_valid_Basic_" + str(model)+"_"+str(task)+ ".csv", header=None)
# shap_values_for_full_pred = pd.read_csv("../Validation_data_for_cards/"+data_version+"_Shap_on_x_valid_" + str(model)+"_"+str(task)+ ".csv", header=None)
MIMIC_preops = pd.read_csv('./MIMIC_preops_ohe_te.csv')


# this is needed because the sens var data is being passsed seperately so not needeed here while testing
if data_version == 'MIMICwithsens':
    y_validation_data = pd.read_csv(
        "../Validation_data_for_cards/MIMICsub_y_true_valid_" + str(model) + "_" + str(task) + ".csv")
    x_validation_data = pd.read_csv(
        "../Validation_data_for_cards/MIMICsub_x_valid_" + str(model) + "_" + str(task) + ".csv")

y_validation_data.rename(columns={'status':task}, inplace=True)
# data saving directory
fig_saving_dir = "./"+str(task)+"_"+str(model)+"_MIMIC_forfigures_in_ModelReportCards_" + str(datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S.%f"))
if not os.path.exists(fig_saving_dir):
    os.makedirs(fig_saving_dir)


sens_var = ['Sex', 'RACE', 'age']



######  Generating full data statistics  ###################

true_outcome_Rate, frequency_level_wise, frequency_level_wise_psm, freq_level_del, sens_var_level, sens_var_level_dict, \
Avg_PPV_Diff, Avg_PPV_Diff_psm, p_PPV_Diff, p_PPV_Diff_psm , \
Avg_Sensitivity_Diff, Avg_Sensitivity_Diff_psm, p_Sensitivity_Diff, p_Sensitivity_Diff_psm, \
Avg_AUROC_Diff, Avg_AUROC_Diff_psm, p_AUROC_Diff, p_AUROC_Diff_psm , Grp_avg_PPV, Grp_avg_PPV_psm, Grp_avg_Sensitivity, Grp_avg_Sensitivity_psm, Grp_avg_AUROC, Grp_avg_AUROC_psm, overall_PPV, overall_Sensitivity , overall_AUROC= Model_Report_card(
    y_validation_pred_full, x_validation_data, y_validation_data, sens_var, task, MIMIC_preops)

Generating_pdf_simple(true_outcome_Rate, frequency_level_wise,
                      frequency_level_wise_psm, freq_level_del)

