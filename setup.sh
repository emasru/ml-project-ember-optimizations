#!/bin/bash
echo 'Activating Conda environment and installing packages'
conda activate ember-env
conda install --file ember/requirements_conda.txt
echo 'Running setup.py file for EMBER'
cd ember
python setup.py install
cd ..
echo 'Done'