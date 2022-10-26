# AlgorithmicBias_Delirium_ML4H22022
Code for ML4H'22 submission

This repository contains the code used in the project investigating algorithmic bias in delirium predicting machine learning models.

We used two datasets: ACTFAST Dataset (Epic era data from Barnes Jewish Hopital in St Louis; not available publicly) and MIMIC-III.

For the MIMIC III dataset, the extraction steps were followed from the supplementary information available in the [paper on labelling delirum](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7941123/).

The psql queries for MIMICIII extraction can be found in MIMIC3-delirium_extraction.txt file.

For processing the datasets, partitioning into train-test and training the RF (others models were also tried), we used the file Delirium_model_fairness_testing.py and Delirium_model_fairness_testing_MIMIC.py for EHR_Prop (Epic) and MIMIC-III respectively.

For computing the groupwise performance as reported in the paper, we used Differential_perf_generator.py and Differential_per_generator_MIMIC.py. 

Dependencies: Python 3.9.10
