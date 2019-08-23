import os,sys
import pdb
import json
import argparse
import shutil

TASKS = ["AX-b", "AX-g", "BoolQ", "CB", "COPA", "MultiRC", "ReCoRD", "RTE", "WiC"]


def preprocess(task, args, process_func, splits=["train","val","test"]):
    datadir = os.path.join(args.data_dir, task)
    prodir = os.path.join(datadir,"processed")
    if os.path.isdir(prodir):
        shutil.rmtree(prodir)
    os.mkdir(prodir)
    for split in splits:
        input_fname = os.path.join(datadir,split+'.jsonl')
        with open(input_fname) as fin:
            for line in fin:
                sample = json.loads(line.strip())
                inputs, label = process_func(sample, split)
                pdb.set_trace();


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
        preprocess('COPA', args)
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
