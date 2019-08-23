import os,sys
import pdb
import json
import argparse
import shutil

from multiprocessing_bpe_encoder import main as encoder

TASKS = ["AX-b", "AX-g", "BoolQ", "CB", "COPA", "MultiRC", "ReCoRD", "RTE", "WiC"]


def preprocess(task, args, ninputs, process_func, splits=["train","val"]):
    datadir = os.path.join(args.data_dir, task)
    prodir = os.path.join(datadir,"processed")
    if os.path.isdir(prodir):
        shutil.rmtree(prodir)
    os.mkdir(prodir)
    for split in splits:
        inputfiles = []
        for i in range(ninputs):
            inputfiles.append( open(os.path.join(prodir,split+'.raw.input'+str(i)),"w+") )
        label_file = open(os.path.join(prodir,split+'.label'),"w+")
        input_fname = os.path.join(datadir,split+'.jsonl')
        with open(input_fname) as fin:
            for line in fin:
                example = json.loads(line.strip())
                propcessed_set = process_func(example)
                for inputset, label in propcessed_set:
                    for input_sent, inputfile in zip (inputset,inputfiles):
                        inputfile.write(input_sent); inputfile.write('\n')
                    label_file.write(label);   label_file.write('\n')
        for i in inputfiles:
            i.close()
        label_file.close()


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
            label = str(example['label'])
            return [ # one sample
                    [[context, question + ' ' + choice1, question + ' ' + choice2] , label]
                    ]
        preprocess('COPA', args, 3, process_func)
        encoder(["--encoder-json", "encoder.json", "--vocab-bpe", "vocab.bpe", "--inputs" , "", "--outputs", "", "--workers", "60", "--keep-empty" ])
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


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
