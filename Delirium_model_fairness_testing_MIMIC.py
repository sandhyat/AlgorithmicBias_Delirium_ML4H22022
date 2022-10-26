# import json
# import os
# import sys, argparse
# import glob
# import pickle
#
import numpy as np
import pandas as pd
# import math
# from pyarrow import feather  # directly writing import pyarrow didn't work
from sklearn.model_selection import train_test_split
from sklearn import svm, linear_model, model_selection, metrics
from sklearn.metrics import roc_auc_score, average_precision_score
# from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn import ensemble
# import xgboost as xgb
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
# from torch.autograd import Variable
# from torch.nn import functional as F
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, RocCurveDisplay,PrecisionRecallDisplay,confusion_matrix
import datetime
# import shap

# TODO: There is some issue in the psql query as there are intersecting hadmids between the with and without delirium icdcode tables

task = 'postop_del'
data_gen = 0 # 1 only at the time of data processing and split else 0

# reading the data

if data_gen == 1:
    if True:
        """ PRESCRIPTIONS """
        pres_with = pd.read_csv('/mnt/ris/sandhyat/mimic_delirium_data/prescriptions_with.csv')
        pres_WO = pd.read_csv('/mnt/ris/sandhyat/mimic_delirium_data/prescriptions_WO.csv')


        pres_with['drug_type'] = 0
        pres_WO['drug_type'] = 0

        # prescriptions processing
        benz_idx_with = pres_with[pres_with['drug'].str.contains('azepam') == True].index
        pres_with.loc[benz_idx_with, 'drug_type'] = 'Bendodiazepines'

        benz_idx_WO = pres_WO[pres_WO['drug'].str.contains('azepam') == True].index
        pres_WO.loc[benz_idx_WO, 'drug_type'] = 'Bendodiazepines'

        antipsychotics = ["Chlorproma", 'Fluphena', 'Haloperid', 'Loxapi', 'Perphenaz', 'Pimoz', 'Thiroida', 'Thiotix',
                          'Trifluoper', 'Aripipr', 'Asenap','Brexipip', 'Caripra', 'Clozap', 'Iloperi', 'Lurasi','Olanzap',
                          'Paliperi','Pimavan','Quetiap','Resperid','Ziprasi']

        antipsy_idx_with = pres_with[pres_with['drug'].str.contains('|'.join(antipsychotics))==True].index
        pres_with.loc[antipsy_idx_with, 'drug_type'] = 'AntiPsychotics'

        antipsy_idx_WO = pres_WO[pres_WO['drug'].str.contains('|'.join(antipsychotics))==True].index
        pres_WO.loc[antipsy_idx_WO, 'drug_type'] = 'AntiPsychotics'

        riv_idx_with = pres_with[pres_with['drug'].str.contains('Rivastig') == True].index
        pres_with.loc[riv_idx_with, 'drug_type'] = 'Rivastigmine'

        riv_idx_WO = pres_WO[pres_WO['drug'].str.contains('Rivastig') == True].index
        pres_WO.loc[riv_idx_WO, 'drug_type'] = 'Rivastigmine'

        Dex_idx_with = pres_with[pres_with['drug'].str.contains('Dexmede') == True].index
        pres_with.loc[Dex_idx_with, 'drug_type'] = 'Dexmedetomidine'

        Dex_idx_WO = pres_WO[pres_WO['drug'].str.contains('Dexmede') == True].index
        pres_WO.loc[Dex_idx_WO, 'drug_type'] = 'Dexmedetomidine'

        pres_with.drop(pres_with[pres_with['drug_type'] == 0].index, inplace=True)
        pres_WO.drop(pres_WO[pres_WO['drug_type'] == 0].index, inplace=True)

        pres_with.drop(columns=['drug_name_generic', 'drug_name_poe'], inplace=True)
        pres_WO.drop(columns=['drug_name_generic', 'drug_name_poe'], inplace=True)

        pres_with['timedate'] = pd.to_datetime(0)
        pres_with.rename(columns={'startdate': 'date', 'drug_type': 'event', 'drug':'details'}, inplace=True)
        pres_with = pres_with[["hadm_id", "event", "details", "date", "timedate"]]

        pres_WO['timedate'] = pd.to_datetime(0)
        pres_WO.rename(columns={'startdate': 'date', 'drug_type': 'event', 'drug':'details'}, inplace=True)
        pres_WO = pres_WO[["hadm_id", "event", "details", "date", "timedate"]]

        print("prescriptions done")

        """ PROCEDURES (RADIOLOGY  & ECG) """
        rad_with = pd.read_csv('/mnt/ris/sandhyat/mimic_delirium_data/radiology_with.csv')
        rad_WO = pd.read_csv('/mnt/ris/sandhyat/mimic_delirium_data/radiology_WO.csv')
        ecg_with = pd.read_csv('/mnt/ris/sandhyat/mimic_delirium_data/ecg_with.csv')
        ecg_WO = pd.read_csv('/mnt/ris/sandhyat/mimic_delirium_data/ecg_WO.csv')


        chest_des = ["CHEST (PA, LAT & OBLIQUES)" , "CHEST (SINGLE VIEW)", "CHEST (APICAL LORDOTIC ONLY)" , "R RIB UNILAT, W/ AP CHEST RIGHT"  , "CHEST AP ONLY" , "RIB BILAT, W/AP CHEST" , "AP/PA SINGLE VIEW EXPIRATORY CHEST" , "L RIB UNILAT, W/ AP CHEST LEFT"  , "L CHEST (LAT DECUB ONLY) LEFT", "L CHEST (LAT DECUB ONLY) LEFT" , "CHEST (LAT DECUB ONLY)"  , "B CHEST (LAT DECUB ONLY) BILAT", "R CHEST (LAT DECUB ONLY) RIGHT", "CHEST (PA & LAT)" , "CHEST (PORTABLE AP)" ]
        head_des = ["MR HEAD W & W/O CONTRAST"  , "BRAIN SCAN"    , "CT HEAD W/O CONTRAST", "MR HEAD W/O CONTRAST",  "CT EMERGENCY HEAD W/O CONTRAST", "CT HEAD W/ CONTRAST", "CT HEAD W/ & W/O CONTRAST", "MR HEAD W/ CONTRAST", "CT HEAD W/ ANESTHESIA W/ CONTRAST", "PORTABLE HEAD CT W/O CONTRAST"]

        rad_with['proc_type']=0
        rad_WO['proc_type']=0

        chest_idx_with = rad_with[rad_with['description'].apply(lambda  x: any([k in x for k in chest_des]))].index
        rad_with.loc[chest_idx_with, 'proc_type'] = 'CXR'

        chest_idx_WO = rad_WO[rad_WO['description'].apply(lambda  x: any([k in x for k in chest_des]))].index
        rad_WO.loc[chest_idx_WO, 'proc_type'] = 'CXR'

        head_idx_with = rad_with[rad_with['description'].apply(lambda x: any([k in x for k in head_des]))].index
        rad_with.loc[head_idx_with, 'proc_type'] = 'BrainImaging'

        head_idx_WO = rad_WO[rad_WO['description'].apply(lambda x: any([k in x for k in head_des]))].index
        rad_WO.loc[head_idx_WO, 'proc_type'] = 'BrainImaging'

        rad_with.drop(rad_with[rad_with['proc_type'] == 0].index, inplace=True)
        rad_WO.drop(rad_WO[rad_WO['proc_type'] == 0].index, inplace=True)

        ecg_with['proc_type'] = ecg_with['category']
        ecg_WO['proc_type'] = ecg_WO['category']

        procedures_with = pd.concat([rad_with, ecg_with])
        procedures_WO = pd.concat([rad_WO, ecg_WO])

        procedures_with.drop(columns=['category', 'cgid'], inplace=True)
        procedures_WO.drop(columns=['category', 'cgid'], inplace=True)

        procedures_with['timedate'] = pd.to_datetime(0)
        procedures_with.rename(columns={'chartdate': 'date', 'proc_type': 'event', 'description':'details'}, inplace=True)
        procedures_with = procedures_with[["hadm_id", "event", "details", "date", "timedate"]]

        procedures_WO['timedate'] = pd.to_datetime(0)
        procedures_WO.rename(columns={'chartdate': 'date', 'proc_type': 'event', 'description':'details'}, inplace=True)
        procedures_WO = procedures_WO[["hadm_id", "event", "details", "date", "timedate"]]

        print("procedures done")

        """  LABS """

        labs_with = pd.read_csv('/mnt/ris/sandhyat/mimic_delirium_data/labs_with.csv')
        labs_WO = pd.read_csv('/mnt/ris/sandhyat/mimic_delirium_data/labs_WO.csv')

        # this is done to make sure the issue of iterating over float doesn't arise as nan is a float
        labs_with = labs_with.iloc[labs_with['label'].dropna().index]
        labs_WO = labs_WO.iloc[labs_WO['label'].dropna().index]

        cbc = ["Red Blood Cells", "White Blood Cells", "RBC", "WBC", "Platelet Count", "Basophils", "Neutrophils", "Lymphocytes", "Monocytes", "Bands", "Macrocytes", "Absolute Lymphocyte Count", "Lymphocytes, Percent", "Reticulocyte Count, Automated", "Granulocyte Count", "Macrophages", "WBC Count", "Eosinophil Count", "Monocyte Count"]
        elec = ["Calcium, Total", "Magnesium", "Phosphate", "Potassium", "Sodium", "Free Calcium"]
        liver = ["Alanine Aminotransferase (ALT)", "Albumin", "Asparate Aminotransferase (AST)", "<Albumin>"]
        renal = ["Creatinine", "Urea Nitrogen", "Estimated GFR (MDRD equation)", "Creatinine Clearance", "Creatinine, Serum", "Urine Creatinine", "24 hr Creatinine", "Albumin/Creatinine, Urine", "Alkaline Phosphatase"]
        tox = ["Barbiturate Screen", "Tricyclic Antidepressant Screen", "Barbiturate Screen, Urine", "Cocaine, Urine",
                  "Opiate Screen, Urine", "Acetaminophen", "Benzodiazepine Screen", "Salicylate",
                  "Amphetamine Screen, Urine", "Benzodiazepine Screen, Urine", "Methadone, Urine", "Marijuana"]
        b = ["Vitamin B12", "Folate"]
        t4 = ["Thyroxine (T4), Free", "Thyroid Stimulating Hormone", "Thyroxine (T4)"]
        ai = ["Anti-Nuclear Antibody, Titer", "Anti-Nuclear Antibody", "Anti-Neutrophil Cytoplasmic Antibody"]
        lp = ["Total Protein, CSF", "WBC, CSF", "Glucose, CSF", "RBC, CSF", "Hematocrit, CSF", "Miscellaneous, CSF"]

        labs_with['lab_type'] = 0

        labs_with.loc[labs_with[labs_with['label'].apply(lambda x: any([k in x for k in ["BLOOD GAS", "Blood Gas"]]))].index, 'lab_type'] = 'ABG'
        labs_with.loc[labs_with[labs_with['label'].apply(lambda x: any([k in x for k in cbc]))].index, 'lab_type'] = 'CBC'
        labs_with.loc[labs_with[labs_with['label'].apply(lambda x: any([k in x for k in elec]))].index, 'lab_type'] = 'ElectrolytePanel'
        labs_with.loc[labs_with[labs_with['label'].apply(lambda x: any([k in x for k in liver]))].index, 'lab_type'] = 'LiverFunction'
        labs_with.loc[labs_with[labs_with['label'].apply(lambda x: any([k in x for k in renal]))].index, 'lab_type'] = 'RenalFunction'
        labs_with.loc[labs_with[labs_with['label'].apply(lambda x: any([k in x for k in tox]))].index, 'lab_type'] = 'ToxScreen'
        labs_with.loc[labs_with[labs_with['label'].apply(lambda x: any([k in x for k in b]))].index, 'lab_type'] = 'Bvitamins'
        labs_with.loc[labs_with[labs_with['label'].apply(lambda x: any([k in x for k in t4]))].index, 'lab_type'] = 'ThyroidFunction'
        labs_with.loc[labs_with[labs_with['label'].apply(lambda x: any([k in x for k in ['Cortisol']]))].index, 'lab_type'] = 'Cortisol'
        labs_with.loc[labs_with[labs_with['label'].apply(lambda x: any([k in x for k in ['Sedimentation Rate']]))].index, 'lab_type'] = 'SedimentationRate'
        labs_with.loc[labs_with[labs_with['label'].apply(lambda x: any([k in x for k in ['Ammonia']]))].index, 'lab_type'] = 'Ammonia'
        labs_with.loc[labs_with[labs_with['label'].apply(lambda x: any([k in x for k in ai]))].index, 'lab_type'] = 'AutoimmuneSerology'
        labs_with.loc[labs_with[labs_with['label'].apply(lambda x: any([k in x for k in ['HIV Antibody']]))].index, 'lab_type'] = 'HIVantibody'
        labs_with.loc[labs_with[labs_with['label'].apply(lambda x: any([k in x for k in lp]))].index, 'lab_type'] = 'LumbarPunctureLabs'

        labs_WO['lab_type'] = 0

        labs_WO.loc[labs_WO[labs_WO['label'].apply(lambda x: any([k in x for k in ["BLOOD GAS", "Blood Gas"]]))].index, 'lab_type'] = 'ABG'
        labs_WO.loc[labs_WO[labs_WO['label'].apply(lambda x: any([k in x for k in cbc]))].index, 'lab_type'] = 'CBC'
        labs_WO.loc[labs_WO[labs_WO['label'].apply(lambda x: any([k in x for k in elec]))].index, 'lab_type'] = 'ElectrolytePanel'
        labs_WO.loc[labs_WO[labs_WO['label'].apply(lambda x: any([k in x for k in liver]))].index, 'lab_type'] = 'LiverFunction'
        labs_WO.loc[labs_WO[labs_WO['label'].apply(lambda x: any([k in x for k in renal]))].index, 'lab_type'] = 'RenalFunction'
        labs_WO.loc[labs_WO[labs_WO['label'].apply(lambda x: any([k in x for k in tox]))].index, 'lab_type'] = 'ToxScreen'
        labs_WO.loc[labs_WO[labs_WO['label'].apply(lambda x: any([k in x for k in b]))].index, 'lab_type'] = 'Bvitamins'
        labs_WO.loc[labs_WO[labs_WO['label'].apply(lambda x: any([k in x for k in t4]))].index, 'lab_type'] = 'ThyroidFunction'
        labs_WO.loc[labs_WO[labs_WO['label'].apply(lambda x: any([k in x for k in ['Cortisol']]))].index, 'lab_type'] = 'Cortisol'
        labs_WO.loc[labs_WO[labs_WO['label'].apply(lambda x: any([k in x for k in ['Sedimentation Rate']]))].index, 'lab_type'] = 'SedimentationRate'
        labs_WO.loc[labs_WO[labs_WO['label'].apply(lambda x: any([k in x for k in ['Ammonia']]))].index, 'lab_type'] = 'Ammonia'
        labs_WO.loc[labs_WO[labs_WO['label'].apply(lambda x: any([k in x for k in ai]))].index, 'lab_type'] = 'AutoimmuneSerology'
        labs_WO.loc[labs_WO[labs_WO['label'].apply(lambda x: any([k in x for k in ['HIV Antibody']]))].index, 'lab_type'] = 'HIVantibody'
        labs_WO.loc[labs_WO[labs_WO['label'].apply(lambda x: any([k in x for k in lp]))].index, 'lab_type'] = 'LumbarPunctureLabs'

        labs_with.drop(labs_with[labs_with['lab_type'] == 0].index, inplace=True)
        labs_WO.drop(labs_WO[labs_WO['lab_type'] == 0].index, inplace=True)


        labs_with.drop(columns=['fluid', 'value', 'valuenum', 'category'], inplace=True)
        labs_WO.drop(columns=['fluid', 'value', 'valuenum', 'category'], inplace=True)

        labs_with['date'] = pd.to_datetime(0)
        labs_with.rename(columns={'charttime': 'timedate', 'lab_type': 'event', 'label':'details'}, inplace=True)
        labs_with = labs_with[["hadm_id", "event", "details", "date", "timedate"]]

        labs_WO['date'] = pd.to_datetime(0)
        labs_WO.rename(columns={'charttime': 'timedate', 'lab_type': 'event', 'label':'details'}, inplace=True)
        labs_WO = labs_WO[["hadm_id", "event", "details", "date", "timedate"]]

        print("labs done")

        """  MICROBIOLOGY """

        microbiology_with = pd.read_csv('/mnt/ris/sandhyat/mimic_delirium_data/microbiology_with.csv')
        microbiology_WO = pd.read_csv('/mnt/ris/sandhyat/mimic_delirium_data/microbiology_WO.csv')

        # this is done to make sure the issue of iterating over float doesn't arise as nan is a float
        microbiology_with = microbiology_with.iloc[microbiology_with['spec_type_desc'].dropna().index]
        microbiology_WO = microbiology_WO.iloc[microbiology_WO['spec_type_desc'].dropna().index]


        microbiology_with['lab_type'] = 0
        microbiology_WO['lab_type'] = 0

        blood_cul = ["BLOOD CULTURE", "SEROLOGY/BLOOD", "Blood (Toxo)", "Blood (EBV)", "Blood (CMV AB)"]
        urine_cul = ["URINE", "URINE,SUPRAPUBIC ASPIRATE", "URINE,PROSTATIC MASSAGE", "URINE,KIDNEY"]
        lp_micro = ["CSF;SPINAL FLUID"]

        microbiology_with.loc[microbiology_with[microbiology_with['spec_type_desc'].apply(lambda x: any([k in x for k in blood_cul]))].index, 'lab_type'] = 'BloodCulture'
        microbiology_with.loc[microbiology_with[microbiology_with['spec_type_desc'].apply(lambda x: any([k in x for k in urine_cul]))].index, 'lab_type'] = 'UrineCulture'
        microbiology_with.loc[microbiology_with[microbiology_with['spec_type_desc'].apply(lambda x: any([k in x for k in lp_micro]))].index, 'lab_type'] = 'LumbarPuncture'

        microbiology_WO.loc[microbiology_WO[microbiology_WO['spec_type_desc'].apply(lambda x: any([k in x for k in blood_cul]))].index, 'lab_type'] = 'BloodCulture'
        microbiology_WO.loc[microbiology_WO[microbiology_WO['spec_type_desc'].apply(lambda x: any([k in x for k in urine_cul]))].index, 'lab_type'] = 'UrineCulture'
        microbiology_WO.loc[microbiology_WO[microbiology_WO['spec_type_desc'].apply(lambda x: any([k in x for k in lp_micro]))].index, 'lab_type'] = 'LumbarPuncture'

        microbiology_with.drop(columns=['spec_itemid', 'org_name'], inplace=True)
        microbiology_WO.drop(columns=['spec_itemid', 'org_name'], inplace=True)

        microbiology_with.drop(microbiology_with[microbiology_with['lab_type'] == 0].index, inplace=True)
        microbiology_WO.drop(microbiology_WO[microbiology_WO['lab_type'] == 0].index, inplace=True)

        microbiology_with.drop_duplicates(subset=['hadm_id', 'chartdate', 'lab_type'], inplace=True)
        microbiology_WO.drop_duplicates(subset=['hadm_id', 'chartdate', 'lab_type'], inplace=True)

        microbiology_with.rename(columns={'chartdate': 'date', 'lab_type': 'event', 'charttime':'timedate', 'spec_type_desc':'details'}, inplace=True)
        microbiology_with = microbiology_with[["hadm_id", "event", "details", "date", "timedate"]]

        microbiology_WO.rename(columns={'chartdate': 'date', 'lab_type': 'event', 'charttime':'timedate', 'spec_type_desc':'details'}, inplace=True)
        microbiology_WO = microbiology_WO[["hadm_id", "event", "details", "date", "timedate"]]

        print("micro done")

        """ WORDS WITH HIGH PPV FOR DELIRIUM """

        words_with = pd.read_csv('/mnt/ris/sandhyat/mimic_delirium_data/words_with.csv')
        words_WO = pd.read_csv('/mnt/ris/sandhyat/mimic_delirium_data/words_WO.csv')

        col_names_words = ["hadm_id", "cgid", "chartdate", "ams", "mentalstatus", "deliri", "hallucin", "confus", "reorient", "disorient", "encephalopathy"]

        words_with.rename(columns=dict(zip(words_with.columns, col_names_words)), inplace=True)
        words_WO.rename(columns=dict(zip(words_WO.columns, col_names_words)), inplace=True)

        ams = words_with[['hadm_id', 'chartdate', 'ams']].iloc[words_with['ams'].dropna().index]
        ams['ams'] = "AMS"
        ams.rename(columns = {'ams':"Event"}, inplace=True)

        ms = words_with[['hadm_id', 'chartdate', 'mentalstatus']].iloc[words_with['mentalstatus'].dropna().index]
        ms['mentalstatus'] = "MentalStatus"
        ms.rename(columns = {'mentalstatus':"Event"}, inplace=True)

        deliri = words_with[['hadm_id', 'chartdate', 'deliri']].iloc[words_with['deliri'].dropna().index]
        deliri['deliri'] = "Deliri"
        deliri.rename(columns = {'deliri':"Event"}, inplace=True)

        hallucin = words_with[['hadm_id', 'chartdate', 'hallucin']].iloc[words_with['hallucin'].dropna().index]
        hallucin['hallucin'] = "Hallucin"
        hallucin.rename(columns = {'hallucin':"Event"}, inplace=True)

        confus = words_with[['hadm_id', 'chartdate', 'confus']].iloc[words_with['confus'].dropna().index]
        confus['confus'] = "Confus"
        confus.rename(columns = {'confus':"Event"}, inplace=True)

        reorient = words_with[['hadm_id', 'chartdate', 'reorient']].iloc[words_with['reorient'].dropna().index]
        reorient['reorient'] = "REorient"
        reorient.rename(columns = {'reorient':"Event"}, inplace=True)

        disorient = words_with[['hadm_id', 'chartdate', 'disorient']].iloc[words_with['disorient'].dropna().index]
        disorient['disorient'] = "DISorient"
        disorient.rename(columns = {'disorient':"Event"}, inplace=True)

        encephalopathy = words_with[['hadm_id', 'chartdate', 'encephalopathy']].iloc[words_with['encephalopathy'].dropna().index]
        encephalopathy['encephalopathy'] = "Encephalopathy"
        encephalopathy.rename(columns = {'encephalopathy':"Event"}, inplace=True)

        proc_words_with = pd.concat([ams, ms, deliri, hallucin, confus, reorient, disorient, encephalopathy])
        proc_words_with['details'] = 0
        proc_words_with['timedate'] = pd.to_datetime(0)
        proc_words_with.rename(columns={'chartdate':'date', 'Event':'event'}, inplace =True)
        proc_words_with = proc_words_with[["hadm_id", "event", "details", "date", "timedate"]]


        ams_WO = words_WO[['hadm_id', 'chartdate', 'ams']].iloc[words_WO['ams'].dropna().index]
        ams_WO['ams'] = "AMS"
        ams_WO.rename(columns={'ams': "Event"}, inplace=True)

        ms_WO = words_WO[['hadm_id', 'chartdate', 'mentalstatus']].iloc[words_WO['mentalstatus'].dropna().index]
        ms_WO['mentalstatus'] = "MentalStatus"
        ms_WO.rename(columns={'mentalstatus': "Event"}, inplace=True)

        deliri_WO = words_WO[['hadm_id', 'chartdate', 'deliri']].iloc[words_WO['deliri'].dropna().index]
        deliri_WO['deliri'] = "Deliri"
        deliri_WO.rename(columns={'deliri': "Event"}, inplace=True)

        hallucin_WO = words_WO[['hadm_id', 'chartdate', 'hallucin']].iloc[words_WO['hallucin'].dropna().index]
        hallucin_WO['hallucin'] = "Hallucin"
        hallucin_WO.rename(columns={'hallucin': "Event"}, inplace=True)

        confus_WO = words_WO[['hadm_id', 'chartdate', 'confus']].iloc[words_WO['confus'].dropna().index]
        confus_WO['confus'] = "Confus"
        confus_WO.rename(columns={'confus': "Event"}, inplace=True)

        reorient_WO = words_WO[['hadm_id', 'chartdate', 'reorient']].iloc[words_WO['reorient'].dropna().index]
        reorient_WO['reorient'] = "REorient"
        reorient_WO.rename(columns={'reorient': "Event"}, inplace=True)

        disorient_WO = words_WO[['hadm_id', 'chartdate', 'disorient']].iloc[words_WO['disorient'].dropna().index]
        disorient_WO['disorient'] = "DISorient"
        disorient_WO.rename(columns={'disorient': "Event"}, inplace=True)

        encephalopathy_WO = words_WO[['hadm_id', 'chartdate', 'encephalopathy']].iloc[
            words_WO['encephalopathy'].dropna().index]
        encephalopathy_WO['encephalopathy'] = "Encephalopathy"
        encephalopathy_WO.rename(columns={'encephalopathy': "Event"}, inplace=True)

        proc_words_WO = pd.concat([ams_WO, ms_WO, deliri_WO, hallucin_WO, confus_WO, reorient_WO, disorient_WO, encephalopathy_WO])
        proc_words_WO['details'] = 0
        proc_words_WO['timedate'] = pd.to_datetime(0)
        proc_words_WO.rename(columns={'chartdate': 'date', 'Event': 'event'}, inplace=True)
        proc_words_WO = proc_words_WO[["hadm_id", "event", "details", "date", "timedate"]]

        print("words done")

        combined_table_with = pd.concat([pres_with, procedures_with, labs_with, microbiology_with, proc_words_with])
        # combined_table_with = combined_table_with.drop(columns = ['details'])

        combined_table_WO = pd.concat([pres_WO, procedures_WO, labs_WO, microbiology_WO, proc_words_WO])
        # combined_table_WO = combined_table_WO.drop(columns = ['details'])

        combined_table_with = combined_table_with.drop_duplicates()
        combined_table_WO = combined_table_WO.drop_duplicates()

        combined_table_with.to_csv('./combined_MIMIC_table_with0.csv', index = False)
        combined_table_WO.to_csv('./combined_MIMIC_table_WO0.csv', index = False)

        raw_dems = pd.read_csv('/mnt/ris/sandhyat/mimic_delirium_data/dems.csv')

        raw_dems['admittime'] = pd.to_datetime(raw_dems['admittime']).dt.date
        raw_dems['dob'] = pd.to_datetime(raw_dems['dob']).dt.date
        raw_dems['age'] = raw_dems.apply(lambda e: (e['admittime'] - e['dob']).days / 365, axis=1)

        raw_dems.loc[raw_dems[raw_dems['age'] > 89].index, 'age'] = 90

        raw_dems['LOS'] = (
            (pd.to_datetime(raw_dems['dischtime']) - pd.to_datetime(raw_dems['admittime'])).astype('timedelta64[D]'))
        raw_dems.drop(
            columns=['los', 'icd9_code', 'short_title', 'drg_code', 'drg_mortality', 'drg_severity', 'drg_type',
                     'description'], inplace=True)

        raw_dems['admission_type'] = raw_dems['admission_type'].astype('category')
        raw_dems['admission_location'] = raw_dems['admission_location'].astype('category')
        raw_dems['discharge_location'] = raw_dems['discharge_location'].astype('category')
        raw_dems['insurance'] = raw_dems['insurance'].astype('category')
        raw_dems['religion'] = raw_dems['religion'].astype('category')
        raw_dems['ethnicity'] = raw_dems['ethnicity'].astype('category')
        raw_dems['gender'] = raw_dems['gender'].astype('category')

        raw_dems.drop_duplicates(inplace=True)

        criteria_h = raw_dems.loc[raw_dems[(raw_dems['LOS'] < 31) & (raw_dems['age'] > 18)].index]['hadm_id']

        raw_dems[raw_dems['hadm_id'].isin(criteria_h)].to_csv('./proc_dems.csv', index=False)

        """ back to the combined data"""

        # index where time and date time needs to be merged; there are explicit nans and there are values set to the start of 1970 in python
        idx_time_with = combined_table_with[combined_table_with['timedate'].isin(
            [combined_table_with['timedate'].sort_values()[combined_table_with['timedate'].sort_values().index[-1]],
             combined_table_with['timedate'].sort_values()[
                 combined_table_with['timedate'].sort_values().index[0]]])].index

        # Set the date from GMT to EDT; add 12 hours to set to noon.
        combined_table_with.loc[idx_time_with, 'timedate'] = \
        (pd.to_datetime(combined_table_with['date']) + datetime.timedelta(seconds=57600)).loc[idx_time_with]
        combined_table_with.drop(columns=['date'], inplace=True)
        combined_table_with.dropna(inplace=True)  # there as one wierd entry

        # index where time and date time needs to be merged; there are explicit nans and there are values set to the start of 1970 in python
        idx_time_WO = combined_table_WO[combined_table_WO['timedate'].isin(
            [combined_table_WO['timedate'].sort_values()[combined_table_WO['timedate'].sort_values().index[-1]],
             combined_table_WO['timedate'].sort_values()[
                 combined_table_WO['timedate'].sort_values().index[0]]])].index

        # Set the date from GMT to EDT; add 12 hours to set to noon.
        combined_table_WO.loc[idx_time_WO, 'timedate'] = \
            (pd.to_datetime(combined_table_WO['date']) + datetime.timedelta(seconds=57600)).loc[idx_time_WO]
        combined_table_WO.drop(columns=['date'], inplace=True)
        combined_table_WO.dropna(inplace=True)

        combined_table_with.to_csv('./combined_MIMIC_table_with.csv', index = False)
        combined_table_WO.to_csv('./combined_MIMIC_table_WO.csv', index = False)

        print("combined data saving done")

        # outcome files
        del_pos = pd.read_csv('./positive_del.csv')
        del_neg = pd.read_csv('./negative_del.csv')

        print("combined data reading done")

        pos_criteria_h = set(del_pos['hadm_id'].unique()).intersection(set(criteria_h))
        neg_criteria_h = set(del_neg['hadm_id'].unique()).intersection(set(criteria_h))
        neg_criteria_h = neg_criteria_h.difference(pos_criteria_h)  # some issue with the query

        combined_table_with['temp_col'] = 1
        tab_with = combined_table_with.drop(columns=['details', 'timedate']).groupby(by=['hadm_id', 'event']).sum()
        tab_with = tab_with.reset_index()
        count_matrix_with = tab_with.pivot(index='hadm_id', columns='event', values='temp_col')
        count_matrix_with = count_matrix_with.loc[count_matrix_with.index.intersection(pos_criteria_h)]
        count_matrix_with['status'] = 1
        count_matrix_with.replace(np.nan, 0, inplace=True)

        combined_table_WO['temp_col'] = 1
        tab_WO = combined_table_WO.drop(columns=['details', 'timedate']).groupby(by=['hadm_id', 'event']).sum()
        tab_WO = tab_WO.reset_index()
        count_matrix_WO = tab_WO.pivot(index='hadm_id', columns='event', values='temp_col')
        count_matrix_WO = count_matrix_WO.loc[count_matrix_WO.index.intersection(neg_criteria_h)]
        count_matrix_WO['status'] = 0
        count_matrix_WO.replace(np.nan, 0, inplace=True)

        count_matrix = pd.concat([count_matrix_with, count_matrix_WO])

        count_matrix['LumbarPuncture'] = count_matrix['LumbarPuncture'] + count_matrix['LumbarPunctureLabs']
        count_matrix.drop(columns=['LumbarPunctureLabs'], inplace=True)

        count_matrix_tr, count_matrix_te = train_test_split(count_matrix, test_size=0.3, random_state=42,
                                                            stratify=count_matrix['status'])

        print("count matrix done")

        count_matrix_tr.to_csv('./count_matrix_tr.csv', index=True)
        count_matrix_te.to_csv('./count_matrix_te.csv', index=True)
else:
    count_matrix_tr = pd.read_csv('./count_matrix_tr.csv')
    count_matrix_te = pd.read_csv('./count_matrix_te.csv')

    count_matrix_tr.set_index('hadm_id', inplace=True)
    count_matrix_te.set_index('hadm_id', inplace=True)

    proc_dems = pd.read_csv('./proc_dems.csv')

    proc_dems_subsfeat = proc_dems[['hadm_id', 'age', 'gender', 'ethnicity', 'insurance']]

    small_ethnic_groups = proc_dems_subsfeat[proc_dems_subsfeat['ethnicity'].isin(['WHITE', 'BLACK/AFRICAN AMERICAN', 'HISPANIC OR LATINO'])]

    # renaming so that the existing code that was used for epic dataset can be used here too
    small_ethnic_groups.rename(columns={'gender':'Sex', 'ethnicity':'RACE'}, inplace=True)

    # rescaling the sex variable
    small_ethnic_groups.loc[(small_ethnic_groups.Sex == 'M'), 'Sex'] = 0
    small_ethnic_groups.loc[(small_ethnic_groups.Sex == 'F'), 'Sex'] = 1

    small_ethnic_groups.loc[(small_ethnic_groups.RACE == 'WHITE'), 'RACE'] = 0
    small_ethnic_groups.loc[(small_ethnic_groups.RACE == 'BLACK/AFRICAN AMERICAN'), 'RACE'] = 1
    small_ethnic_groups.loc[(small_ethnic_groups.RACE == 'HISPANIC OR LATINO'), 'RACE'] = -1

    # this is done so that only few subgroups are included during evalluation
    count_matrix_tr = count_matrix_tr.loc[
        list(set(count_matrix_tr.index).intersection(small_ethnic_groups['hadm_id']))]
    count_matrix_te = count_matrix_te.loc[list(set(count_matrix_te.index).intersection(small_ethnic_groups['hadm_id']))]


    categorical_variables = ['Sex', 'RACE', 'insurance']

    preops_ohe = small_ethnic_groups.copy()
    preops_ohe.drop(columns=categorical_variables, inplace=True)
    import itertools
    encoded_variables = list()
    for i in categorical_variables:
        temp = pd.get_dummies(small_ethnic_groups[i], prefix=i)
        preops_ohe = pd.concat([preops_ohe, temp], axis=1)
        encoded_variables.append([column for column in temp.columns])
    encoded_variables = list(itertools.chain.from_iterable(encoded_variables))

    # this was done to pass the demographics for grouping later
    preops_ohe_te = preops_ohe[preops_ohe['hadm_id'].isin(count_matrix_te.index)]
    preops_ohe_te.to_csv('./MIMIC_preops_ohe_te.csv', index=False)


    criteria_h = set(proc_dems['hadm_id'].unique())

    # # outcome files
    # del_pos = pd.read_csv('./positive_del.csv')
    # del_neg = pd.read_csv('./negative_del.csv')
    #
    # print("combined data reading done")
    #
    # pos_criteria_h = set(del_pos['hadm_id'].unique()).intersection(set(criteria_h))
    # neg_criteria_h = set(del_neg['hadm_id'].unique()).intersection(set(criteria_h))
    # neg_criteria_h = neg_criteria_h.difference(pos_criteria_h) # some issue with the query


    # once you already have the data, the following is done to train the full and basic models on the saved train and validation data
    model_list = ['LR', 'RF', 'DT', 'GBT', 'DNN']

    if False:
        count_matrix_tr = count_matrix_tr.loc[
            list(set(count_matrix_tr.index).intersection(small_ethnic_groups['hadm_id']))]
        preops_ohe_tr = preops_ohe[preops_ohe['hadm_id'].isin(count_matrix_tr.index)]
        preops_ohe_tr.set_index(['hadm_id'], inplace=True)
        y_tr = count_matrix_tr.iloc[:, -1].to_frame().values
        count_matrix_tr = pd.concat([count_matrix_tr.iloc[:,:-1], preops_ohe_tr.iloc[:, 1:]], axis=1)

        y_te = count_matrix_te.iloc[:, -1].to_frame().values
        preops_ohe_te.set_index(['hadm_id'], inplace=True)
        count_matrix_te = pd.concat([count_matrix_te.iloc[:,:-1], preops_ohe_te.iloc[:, 1:]], axis=1)


        for model in model_list:
            if model == "LR":
                clf = linear_model.LogisticRegression(penalty='l1', n_jobs=-1, solver='saga')
            if model in ['RF']:
                clf = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1)
            if model in ["DT"]:
                clf = tree.DecisionTreeClassifier(max_depth=20)
            if model in ['GBT']:
                clf = ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=1)
            if model in ['DNN']:
                clf = MLPClassifier(random_state=1, hidden_layer_sizes=(256, 256), activation='relu', solver='sgd',
                                    alpha=8e-5, batch_size=64, learning_rate_init=0.01, max_iter=50)

            clf.fit(count_matrix_tr.values,
                    y_tr)  # training with sensitive variable
            y_hat_valid = clf.predict_proba(count_matrix_te)
            y_hat_valid = y_hat_valid[:, 1]

            # model performance on the validation set
            print("Model performance on the validation set for Delirium outcome with model: ", model)
            print("auroc: ", roc_auc_score(y_te, y_hat_valid))
            print("auprc: ", average_precision_score(y_te, y_hat_valid))

            count_matrix_te.iloc[:, :-1].to_csv(
                "../Validation_data_for_cards/MIMICwithsens_x_valid_" + str(model) + "_" + str(task) + ".csv")
            count_matrix_te.iloc[:, -1].to_csv(
                "../Validation_data_for_cards/MIMICwithsens_y_true_valid_" + str(model) + "_" + str(task) + ".csv")
            np.savetxt(
                "../Validation_data_for_cards/MIMICwithsens_y_pred_prob_valid_Full_" + str(model) + "_" + str(task) + ".csv",
                y_hat_valid, delimiter=",")


    """ FULL MODEL"""
    for model in model_list:
        if model == "LR":
            clf = linear_model.LogisticRegression(penalty='l1', n_jobs=-1, solver='saga')
        if model in ['RF']:
            clf = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1)
        if model in ["DT"]:
            clf = tree.DecisionTreeClassifier(max_depth=20)
        if model in ['GBT']:
            clf = ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=1)
        if model in ['DNN']:
            clf = MLPClassifier(random_state=1, hidden_layer_sizes=(256, 256), activation='relu', solver='sgd',
                                alpha=8e-5, batch_size=64, learning_rate_init=0.01, max_iter=50)

        clf.fit(count_matrix_tr.iloc[:, :-1].values,
                count_matrix_tr.iloc[:, -1].to_frame().values)  # training with sensitive variable
        y_hat_valid = clf.predict_proba(count_matrix_te.iloc[:, :-1].values)
        y_hat_valid = y_hat_valid[:, 1]

        # model performance on the validation set
        print("Model performance on the validation set for Delirium outcome with model: ", model)
        print("auroc: ", roc_auc_score(count_matrix_te.iloc[:, -1].to_frame().values, y_hat_valid))
        print("auprc: ", average_precision_score(count_matrix_te.iloc[:, -1].to_frame().values, y_hat_valid))

        count_matrix_te.iloc[:, :-1].to_csv("../Validation_data_for_cards/MIMICsub_x_valid_" + str(model) + "_" + str(task) + ".csv")
        count_matrix_te.iloc[:, -1].to_csv("../Validation_data_for_cards/MIMICsub_y_true_valid_" + str(model) + "_" + str(task) + ".csv")
        np.savetxt("../Validation_data_for_cards/MIMICsub_y_pred_prob_valid_Full_" + str(model) + "_" + str(task) + ".csv",
                   y_hat_valid, delimiter=",")


print('finished')

