from __future__ import print_function

import argparse
import numpy as np
import os,sys
import csv
from tqdm import tqdm
import ipdb
import math
from pandas import DataFrame
import pandas as pd
from scipy import stats


TASKS = ['CoLA','MNLI','MRPC','QNLI','QQP','RTE','SST-2','STS-B','WSC', "AX", 'MNLI_DEV']
parser = argparse.ArgumentParser(description='Roberta GLUE Ensemble')
parser.add_argument('--task', default='MNLI', type=str, choices=TASKS, help='Task Names')
parser.add_argument('--savedir', default='glue_submission/', type=str, help='submission file save dir')

def ensemble_results(taskdir, filename):
    subdirs = os.listdir(taskdir)
    df_main = None
    for i,d in enumerate(subdirs):
        subdir = os.path.join(taskdir, d)
        if not os.path.isdir(subdir):
            continue
        tsvfile=os.path.join(subdir,filename)
        df = pd.read_csv(tsvfile, sep="\t", header=0)
        if df_main is None:
            df_main = df
        else:
            df_main = pd.merge(df_main, df, on='index', how='right', suffixes=("", '_{}'.format(i)))
    all_results = df_main.drop(['index'], axis=1).to_numpy()
    ensemble_result = stats.mode(all_results, axis=1).mode[:,0]
    with open(os.path.join(taskdir, filename), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['index', 'prediction'])
        for i,l in enumerate(ensemble_result):
            tsv_writer.writerow([i, l])

if __name__ == '__main__':
    args = parser.parse_args()
    print("=========== {} ===========".format(args.task))
    if args.task == 'MNLI':
        ensemble_results(os.path.join(args.savedir,args.task), 'MNLI-m.tsv')
        ensemble_results(os.path.join(args.savedir,args.task), 'MNLI-mm.tsv')
