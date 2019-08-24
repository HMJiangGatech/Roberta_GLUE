import os,sys
import pdb
import json
import argparse
import shutil
from tqdm import tqdm
from fairseq import options

from multiprocessing_bpe_encoder import main as encoder

TASKS = ["AX-b", "AX-g", "BoolQ", "CB", "COPA", "MultiRC", "ReCoRD", "RTE", "WiC"]


def binarize(arguments):
    from preprocess import main as binarize_func
    parser = options.get_preprocessing_parser()
    args = parser.parse_args(arguments)
    binarize_func(args)

def preprocess(task, args, ninputs, process_func, splits=["train","val","test"],
                train_splits=["train"], val_splits=["val"], test_splits=["test"]):
    print("========Preprocess {}========".format(task))
    datadir = os.path.join(args.data_dir, task)
    prodir = os.path.join(datadir,"processed")
    if os.path.isdir(prodir):
        shutil.rmtree(prodir)
    os.mkdir(prodir)

    for split in splits:
        print("Split {}:".format(split))
        inputfiles = []
        for i in range(ninputs):
            inputfiles.append( open(os.path.join(prodir,split+'.raw.input'+str(i)),"w") )
        label_file = open(os.path.join(prodir,split+'.label'),"w")
        input_fname = os.path.join(datadir,split+'.jsonl')
        with open(input_fname) as fin:
            for line in tqdm(fin):
                example = json.loads(line.strip())
                propcessed_set = process_func(example)
                for inputset, label in propcessed_set:
                    for input_sent, inputfile in zip (inputset,inputfiles):
                        inputfile.write(input_sent); inputfile.write('\n')
                    label_file.write(str(label));   label_file.write('\n')
        for i,inputfile in enumerate(inputfiles):
            inputfile.close()
            encoder(["--encoder-json", "encoder.json", "--vocab-bpe", "vocab.bpe",
                    "--inputs" , str(os.path.join(prodir,split+'.raw.input'+str(i))),
                    "--outputs", str(os.path.join(prodir,split+'.input'+str(i))),
                    "--workers", "60", "--keep-empty" ])
        label_file.close()

    if os.path.isdir("{}-bin".format(task)):
        shutil.rmtree("{}-bin".format(task))
    os.mkdir("{}-bin".format(task))

    # Binarize Input
    for i in range(ninputs):
        binarize_args = ["--only-source", ]
        if len(train_splits) > 0:
            binarize_args += ["--trainpref", ','.join([os.path.join(prodir,split+'.input'+str(i)) for split in train_splits])]
        if len(val_splits) > 0:
            binarize_args += ["--validpref", ','.join([os.path.join(prodir,split+'.input'+str(i)) for split in val_splits])]
        if len(test_splits) > 0:
            binarize_args += ["--testpref", ','.join([os.path.join(prodir,split+'.input'+str(i)) for split in test_splits])]
        binarize_args += ["--destdir", "{}-bin/input{}".format(task,i), "--workers", "10", "--srcdict", "dict.txt"]
        binarize(binarize_args)

    # Binarize Label
    binarize_args = ["--only-source", ]
    if len(train_splits) > 0:
        binarize_args += ["--trainpref", ','.join([os.path.join(prodir,split+'.label') for split in train_splits])]
    if len(val_splits) > 0:
        binarize_args += ["--validpref", ','.join([os.path.join(prodir,split+'.label') for split in val_splits])]
    binarize_args += ["--destdir", "{}-bin/label".format(task), "--workers", "10"]
    binarize(binarize_args)


def get_tasks(task_names):
    task_names = task_names.split(',')
    if "all" in task_names:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names:
            assert task_name in TASKS, "Task %s not found!" % task_name
            tasks.append(task_name)
    return tasks

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='directory to save data to', type=str, default='superglue_data')
    parser.add_argument('--tasks', help='tasks to be preprocess',
                        type=str, default='all')
    args = parser.parse_args(arguments)

    assert os.path.isdir(args.data_dir)
    tasks = get_tasks(args.tasks)

    print('Copy preprocess code')
    shutil.copyfile('../fairseq/preprocess.py', './preprocess.py')

    if 'Ax-b' in tasks:
        pass
    if 'Ax-g' in tasks:
        pass
    if 'BoolQ' in tasks:
        pass
    if 'CB' in tasks:
        pass
    if 'COPA' in tasks:
        def process_func(example):
            context = example["premise"]
            choice1 = example["choice1"]
            choice2 = example["choice2"]
            question = example["question"]
            question = (
                    "What was the cause of this?"
                    if question == "cause"
                    else "What happened as a result?"
                )
            try:
                label = str(example['label'])
            except:
                label = -1
            return [ # one sample
                    [[context, question + ' ' + choice1, question + ' ' + choice2] , label]
                    ]
        preprocess('COPA', args, 3, process_func)
    if 'MultiRC' in tasks:
        pass
    if 'ReCoRD' in tasks:
        pass
    if 'RTE' in tasks:
        pass
    if 'WiC' in tasks:
        pass
    if 'WSC' in tasks:
        pass
    os.remove("preprocess.py")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
