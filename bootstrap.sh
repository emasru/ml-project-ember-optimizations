#!/bin/bash
mkdir data
cd data
echo 'Downloading dataset'
wget https://ember.elastic.co/ember_dataset_2018_2.tar.bz2
echo 'Extracting dataset'
tar -xjf ember_dataset_2018_2.tar.bz2
rm ember_dataset_2018_2.tar.bz2
cd ..
echo 'Cloning EMBER git repository for helper functions'
git clone https://github.com/elastic/ember.git
echo 'Creating a Conda environment with Python 3.6.15'
conda config --set auto_activate_base false
conda config --add channels conda-forge
conda create -n "ember-env-test" -c conda-forge python=3.6.15
conda init
echo 'You now have to close your current shell, open a new one and run setup.sh'