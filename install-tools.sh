#!/usr/bin/env bash
# Author: Santiago Smith Silva
# Contact: ssilvar@unal.edu.co
# 2017 - National University of Colombia

# Request permission
sudo su

# Install python tools
apt-get install python-pip python-vtk

# Install python packages
pip install --upgrade pip
pip install setuptools
pip install nipy
pip install dipy
pip install matplotlib

