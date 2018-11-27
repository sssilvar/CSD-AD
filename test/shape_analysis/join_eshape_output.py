#!/bin/env python
__author__ = "Santiago Smith Silva"
__description__ = """
A script to join the output of enigma shape.
It joins these two files from the ENIGMA pipeline.
    - *_LogJacs.csv
    - *_thick.csv
USAGE
    join_shape_output.py -f [FOLDER_WITH_ENIGMA_OUTPUT]
"""

import os
import argparse
from os.path import join
from argparse import RawTextHelpFormatter

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__description__, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-f', metavar='--folder',
                        help='Folder to ENIGMA Shape output',
                        required=True)
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Load files
    csv_files = [f for f in os.listdir(args.f) if '_LogJacs.csv' in f or '_thick.csv' in f]
    
    # Load Thickness file and LogJacs
    dfa = pd.read_csv(join(args.f, csv_files[0]), index_col=0, low_memory=False)
    dfb = pd.read_csv(join(args.f, csv_files[1]), index_col=0, low_memory=False)

    # Concatenate
    df = pd.concat([dfa, dfb], axis=1, ignore_index=False, sort=False)
    
    # Print some info
    print('[  INFO  ] Shape of Thickness data: %s' % str(dfa.shape))
    print('[  INFO  ] Shape of LogJacs data: %s' % str(dfb.shape))
    print('[  INFO  ] Shape of new DataFrame: %s' % str(df.shape))

    df.to_csv(join(args.f, 'groupfile_features.csv'))


