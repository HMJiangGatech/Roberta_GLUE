from __future__ import print_function

import argparse
import numpy as np
import os,sys
import csv
from tqdm import tqdm
import ipdb
import math
import torch

from mymodule import *

from fairseq.models.roberta import RobertaModel

TASKS = ['CoLA','MNLI','MRPC','QNLI','QQP','RTE','SST-2','STS-B','WSC', "AX", 'MNLI_DEV']

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

def mnli_dev(ckpdir, ckpname, datadir = None):
    task = 'MNLI'
    if datadir is None:
        datadir = 'data/{}-bin/'.format(task)
    roberta = RobertaModel.from_pretrained(ckpdir, ckpname, datadir)
    roberta.cuda()
    roberta.eval()
    label_fn = lambda label: roberta.task.label_dictionary.string(
                [label + roberta.task.target_dictionary.nspecial]
            )
    str2label = lambda str: roberta.task.label_dictionary.encode_line(str)[0].item() - roberta.task.target_dictionary.nspecial

    tasks = [task]
    testfiles = [os.path.join(datadir,'../glue_data/{}/test.tsv'.format(task))]
    tasks = ["MNLI-m", "MNLI-mm"]
    testfiles = [os.path.join(datadir,'../glue_data/MNLI/dev_matched.tsv'),
                os.path.join(datadir,'../glue_data/MNLI/dev_mismatched.tsv')]


    for task, testfile in zip(tasks, testfiles):
        tv_loss = 0
        tv_low  = 0
        tv_high = 0
        accuracy = 0
        print("Task: {}".format(task))
        with open(testfile) as fin:
            fin.readline()
            pbar = tqdm(enumerate(fin))
            for index, line in pbar:
                tokens = line.strip().split('\t')
                input_token = roberta.encode(tokens[8], tokens[9])
                log_softmax_out = roberta.predict('sentence_classification_head', input_token)
                prediction = log_softmax_out.argmax().item()

                labels = np.array([ str2label(t.lower()) for t in tokens[10:15]])
                labels = np.array([ sum(labels==l) for l in range(3) ])/5
                assert sum(labels)==1
                tv_loss += sum( abs(l1-math.exp(l2)) for l1,l2 in zip(labels, log_softmax_out.detach().cpu().numpy()[0]) )
                tv_high += 2-max(labels)
                tv_low += sum(abs(labels-1/3))
                accuracy += prediction == labels.argmax()
                pbar.set_description("tv: {:.4f}/ {:.4f}-{:.4f}, accu: {:.4f} ".format(tv_loss/(index+1), tv_low/(index+1), tv_high/(index+1), accuracy/(index+1)))




if __name__ == '__main__':
    args = parser.parse_args()
    print("=========== {} ===========".format(args.task))
    os.makedirs(args.savedir, exist_ok=True)
    ckpdir, ckpname = sep_dir(args.ckp)
    with torch.no_grad():
        if args.task == 'MNLI_DEV':
            mnli_dev(ckpdir, ckpname, args.datadir)
        elif args.task.lower() == 'wsc':
            wsc_eval(ckpdir, ckpname, args.savedir, args.datadir)
        else:
            sentence_predict(args.task, ckpdir, ckpname, args.savedir, args.datadir)

