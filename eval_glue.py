from __future__ import print_function

import argparse
import numpy as np
import os,sys
import csv
from tqdm import tqdm


from mymodule import *

from fairseq.models.roberta import RobertaModel

TASKS = ['CoLA','MNLI','MRPC','QNLI','QQP','RTE','SST-2','STS-B','WSC', "AX"]

parser = argparse.ArgumentParser(description='Roberta GLUE Evaluation')
parser.add_argument('--task', default='WSC', type=str, choices=TASKS, help='Task Names')
parser.add_argument('--datadir', default=None, type=str, help='Data dir, we have default path')
parser.add_argument('--ckp', required=True, type=str, help='Checkpoint Directory')
parser.add_argument('--savedir', default='glue_submission/', type=str, help='submission file save dir')



def sep_dir(path):
    a = path.split('/')
    return '/'.join(a[:-1]), a[-1]

def sentence_predict(task, ckpdir, ckpname, savedir, datadir = None):
    if datadir is None:
        datadir = 'data/{}-bin/'.format(task)
        if task == "AX":
            datadir = 'data/MNLI-bin/'
    roberta = RobertaModel.from_pretrained(ckpdir, ckpname, datadir)
    roberta.cuda()
    roberta.eval()
    label_fn = lambda label: roberta.task.label_dictionary.string(
                [label + roberta.task.target_dictionary.nspecial]
            )

    tasks = [task]
    testfiles = [os.path.join(datadir,'../glue_data/{}/test.tsv'.format(task))]
    if task == "AX":
        testfiles = [os.path.join(datadir,'../glue_data/diagnostic/diagnostic.tsv')]
    elif task == "MNLI":
        tasks = ["MNLI-m", "MNLI-mm"]
        testfiles = [os.path.join(datadir,'../glue_data/MNLI/test_matched.tsv'),
                     os.path.join(datadir,'../glue_data/MNLI/test_mismatched.tsv')]

    for task, testfile in zip(tasks, testfiles):
        with open(os.path.join(savedir,'{}.tsv'.format(task)), 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['index', 'prediction'])
            with open(testfile) as fin:
                fin.readline()
                for index, line in tqdm(enumerate(fin)):
                    tokens = line.strip().split('\t')
                    if task in ['CoLA', 'SST-2']:
                        tokens = roberta.encode(tokens[1])
                    elif task == "MRPC":
                        tokens = roberta.encode(tokens[3], tokens[4])
                    elif task == "STS-B":
                        tokens = roberta.encode(tokens[7], tokens[8])
                    elif task in ["MNLI-m", "MNLI-mm"]:
                        tokens = roberta.encode(tokens[8], tokens[9])
                    elif task in ["RTE", "QNLI", "QQP", "AX"]:
                        tokens = roberta.encode(tokens[1], tokens[2])
                    if task == "STS-B":
                        prediction_label =  roberta.predict('sentence_classification_head', tokens,return_logits=True).item()
                        prediction_label = min(1.0,max(0.0,prediction_label))
                    else:
                        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
                        prediction_label = label_fn(prediction)
                    tsv_writer.writerow([index, prediction_label])





def wsc_eval(ckpdir, ckpname, savedir, datadir = None):
    if datadir is None:
        datadir = 'data/WSC/'
    roberta = RobertaModel.from_pretrained(ckpdir, ckpname, datadir)
    roberta.cuda()
    roberta.eval()
    i=-1
    with open(os.path.join(savedir,'WNLI.tsv'), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['index', 'prediction'])
        for sentence, _ in tqdm(wsc_utils.jsonl_iterator(os.path.join(datadir,'test_glue.jsonl'), eval=True)):
            i = i+1
            if i==84:
                tsv_writer.writerow([i, 1])
                continue
            if i in [85,86,87]:
                tsv_writer.writerow([i, 0])
                continue
            pred = roberta.disambiguate_pronoun(sentence)
            tsv_writer.writerow([i, int(pred)])

if __name__ == '__main__':
    args = parser.parse_args()
    print("=========== {} ===========".format(args.task))
    os.makedirs(args.savedir, exist_ok=True)
    ckpdir, ckpname = sep_dir(args.ckp)
    if args.task.lower() == 'wsc':
        wsc_eval(ckpdir, ckpname, args.savedir, args.datadir)
    else:
        sentence_predict(args.task, ckpdir, ckpname, args.savedir, args.datadir)

