# EMBER dataset optimizations

This repository contains the code used in the submission of the paper "Exploring Optimizations of the EMBER Dataset Baseline Model". The original repository for EMBER can be found [here](https://github.com/elastic/ember).

## Setup

The setup assumes you have Conda installed and access to basic GNU/Linux utilities.  

Two setup scripts are provided to set up the environment to run the models: `bootstrap.sh` which should be run first, and `setup.sh` second. It is encouraged to check what the scripts do before running, if the setup is in any way too invasive to your liking, and to make your own changes if necessary.

## File index

### create-vectorized-dataset.py

This creates the vectorized version of the dataset so it can be used in machine learning models.

### benchmark.py

The benchmark model released with the original paper, unchanged, but with our own discrete classifer at the end.

### optimal-benchmark.py

Benchmark model with optimal hyper-parameters.

### rnn-model.py

Simple Recurrent Neural Network model.

### features-skb.py

Finds the top 100 features as classified by Select K-best using the ANOVA-F value.

### gbdt-top50-features-plot.py

Finds the top 50 features with the most 'gain' in the benchmark model and plots them.

### gbdt-top500-features.py

FInds the top 500 features with the most 'gain' and uses only these features to train a new GBDT.

### sorted-results-for-grid-search.txt

The results for the grid-search performed on the hyper-parameters of the baseline model, using the helper function `optimize_model` in the EMBER source-code.