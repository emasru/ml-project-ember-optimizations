#!/bin/bash
echo 'Activating Conda environment and installing packages'
conda activate ember-env-test
conda install --file ember/requirements_conda.txt
echo 'Running setup.py file for EMBER'
cd ember
python setup.py install
cd ..
# These were packages that were missing after following the instructions (dataclasses), plus extra packages for TensorFlow, matplotlib etc.
echo 'Installing missing packages'
pip install dataclasses tensorflow matplotlib
echo 'Done. Run create-vectorized-dataset.py before training the models'