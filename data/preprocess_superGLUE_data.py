import os,sys
import pdb
import json
import argparse
import shutil
from tqdm import tqdm
from fairseq import options

from multiprocessing_bpe_encoder import main as encoder
from fairseq_cli.preprocess import main as binarize_func

TASKS = ["AX-b", "AX-g", "BoolQ", "CB", "COPA", "MultiRC", "ReCoRD", "RTE" ,"WiC"]


def binarize(arguments):
    parser = options.get_preprocessing_parser()
    args = parser.parse_args(arguments)
    binarize_func(args)

def preprocess(task, args, ninputs, process_func, nspans=0, splits=["train","val","test"],
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
        spanfiles = []
        for i in range(nspans):
            spanfiles.append( open(os.path.join(prodir,split+'.raw.span'+str(i)),"w") )
        label_file = open(os.path.join(prodir,split+'.label'),"w")
        input_fname = os.path.join(datadir,split+'.jsonl')
        with open(input_fname) as fin:
            for line in tqdm(fin):
                example = json.loads(line.strip())
                propcessed_set = process_func(example)
                for ex in propcessed_set:
                    if nspans == 0:
                        inputset, label = ex
                    else:
                        inputset, spanset, label = ex
                        for s, spanfile in zip (spanset,spanfiles):
                            spanfile.write(inputset[s[0]][:s[1]]); spanfile.write('\n')
                            spanfile.write(inputset[s[0]][:s[2]]); spanfile.write('\n')
                    for input_sent, inputfile in zip (inputset,inputfiles):
                            inputfile.write(input_sent); inputfile.write('\n')
                    label_file.write(str(label));   label_file.write('\n')
        for i,inputfile in enumerate(inputfiles):
            inputfile.close()
            print("BPE encoding {}/input{}".format(split,i))
            encoder(["--encoder-json", "encoder.json", "--vocab-bpe", "vocab.bpe",
                    "--inputs" , str(os.path.join(prodir,split+'.raw.input'+str(i))),
                    "--outputs", str(os.path.join(prodir,split+'.input'+str(i))),
                    "--workers", "60", "--keep-empty" ])
        for i,spanfile in enumerate(spanfiles):
            spanfile.close()
            print("BPE encoding {}/input{} for counting position".format(split,i))
            encoder(["--encoder-json", "encoder.json", "--vocab-bpe", "vocab.bpe",
                    "--inputs" , str(os.path.join(prodir,split+'.raw.span'+str(i))),
                    "--outputs", str(os.path.join(prodir,split+'.tmp.span'+str(i))),
                    "--workers", "60", "--keep-empty" ])
            with open(os.path.join(prodir,split+'.span'+str(i)),"w") as fout:
                with open(os.path.join(prodir,split+'.tmp.span'+str(i))) as fin:
                    start_pos = None
                    for line in tqdm(fin):
                        tokens = line.split()
                        if start_pos is None:
                            start_pos = len(tokens)
                        else:
                            fout.write('{}\t{}\n'.format(start_pos, len(tokens)))
                            start_pos = None
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

    # Copy
    os.mkdir("{}-bin/span".format(task))
    for split in train_splits+val_splits+test_splits:
        for i in range(nspans):
            shutil.copyfile(os.path.join(prodir,split+'.span'+str(i)),
                            os.path.join("{}-bin/span".format(task),split+'.span'+str(i)))


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

def checkalignment(context, s,e):
    x = context[s:e]
    try:
        assert  (x.rstrip('.,"\' ;:!?&`_-+=<>[]{}^*/\\|') == x) and \
                (x.lstrip('.,"\' ;:!?&`_-+=<>[]{}^*/\\|') == x) and \
                (not context[s-1].isalpha() if s>0 else True) and \
                (not context[e].isalpha() if e<len(context) else True)
    except:
        if x in ['Google+', 'E!', '-4 Fahrenheit', '\'Chariot\' sculpture',
                    '-11.1C', '\'brother\' Per' , 'A-', '\'framing\' Masih']:
            return

        print('=============\n', context)
        # print('=============')
        print(s, e)
        print(x)
        assert  (x.rstrip('.,"\' ;:!?&`_-+=<>[]{}^*/\\|') == x) and \
                (x.lstrip('.,"\' ;:!?&`_-+=<>[]{}^*/\\|') == x) and \
                (not context[s-1].isalpha() if s>0 else True) and \
                (not context[e].isalpha() if e<len(context) else True)


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', help='tasks to be preprocess',
                        type=str, default='all')
    parser.add_argument('--data_dir', help='directory to save data to', type=str, default='superglue_data')
    args = parser.parse_args(arguments)

    assert os.path.isdir(args.data_dir)
    tasks = get_tasks(args.tasks)
    #
    # print('Copy preprocess code')
    # shutil.copyfile('../fairseq/preprocess.py', './preprocess.py')

    if 'Ax-b' in tasks:
        pass
    if 'Ax-g' in tasks:
        pass
    if 'BoolQ' in tasks:
        def process_func(example):
            try:
                label = str(example['label'])
            except:
                label = -1
            return [ # one sample
                    [[example["passage"], example["question"]] , label]
                    ]
        preprocess('BoolQ', args, 2, process_func)
    if 'CB' in tasks:
        def process_func(example):
            try:
                label = str(example['label'])
            except:
                label = -1
            return [ # one sample
                    [[example["premise"], example["hypothesis"]] , label]
                    ]
        preprocess('CB', args, 2, process_func)
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
        def process_func(example):
            passage = example['passage']
            context = passage['text']
            qa_pairs = passage['questions']

            all_samples = []

            for qa in qa_pairs:
                query = qa['question']
                answers = qa['answers']
                for ans in answers:
                    try:
                        label = ans['label']
                    except:
                        label = -1
                    all_samples.append([[context, query + ' ' + ans['text']] , label])
            return all_samples
        preprocess('MultiRC', args, 2, process_func)
    if 'ReCoRD' in tasks:
        # TODO: need to save span
        raise NotImplementedError("Too many noise")
        def process_func(example):
            context = example['passage']['text']
            qa_pairs = example['qas']

            entities =  example['passage']['entities']
            for e in entities:
                e['end'] += 1
                e['text'] = context[e['start']: e['end']]
                if e['text'].endswith('.') or e['text'].endswith('\'') or e['text'].endswith('!'):
                    e['end'] -= 1
                if e['text'].startswith('.'):
                    e['start'] += 1
                if e['text'] == 're found s':
                    e['start'] -= 30
                    e['end'] -= 32
                if e['start'] == 764 and e['end'] == 771 and e['text'].startswith('derri'):
                    example['passage']['text'] = context.replace(e['text']+'re', 'derriere')
                    context = example['passage']['text']
                    e['end'] +=1
                if e['text'] == 'yrian ':
                    e['start'] -= 1
                    e['end'] -= 1
                if e['text'] == 'l-Assad ':
                    e['start'] -= 1
                    e['end'] -= 1
                if context.startswith("Ronnie and Donnie Galyon may be th") and e['start']<600 and example['idx']==2360:
                    e['start'] -= 3
                    e['end'] -= 3
                if e['start'] == 590 and e['end'] == 605 and e['text'].startswith('Michael Garcia'):
                    context=context[:e['end']-1] + '\''+ context[e['end']:]
                    example['passage']['text'] = context
                    e['end'] -= 1
                if e['text'] == 'arth ':
                    context=' '+ context
                    example['passage']['text'] = context
                if e['text'] in ['\nOntari',' Eart']:
                    e['start'] += 1
                    e['end'] += 1
                if e['text'] in ['man so', 'tle of the Seelow Heights ha']:
                    e['start'] -= 3
                    e['end'] -= 3

                e['text'] = context[e['start']: e['end']]
                # if e['text'] == 'he O':
                #     example['passage']['text'] = context[:e['start']-1]+"   "+context[e['start']-1:]
                #     context = example['passage']['text']
                #     e['text'] = context[e['start']: e['end']]

                try:
                    checkalignment(context, e['start'], e['end'])
                except:
                    import pdb; pdb.set_trace()
            #
            # try:
            #     assert example['idx'] != 1827
            # except:
            #     import pdb; pdb.set_trace()

            all_samples = []

            for qa in qa_pairs:
                query = qa['query']
                try:
                    answers = qa['answers']
                    answers = [an['text'] for an in answers]
                except:
                    pass
                for ent in entities:
                    try:
                        label = str(ent['text'] in answers)
                    except:
                        label = -1
                    new_query = query.replace('@placeholder',ent['text'])
                    query_ent_start = query.find('@placeholder')
                    query_ent_end   = query_ent_start+ent['end']-ent['start']
                    all_samples.append([[context, new_query] ,
                                            [[0, ent['start'],ent['end']] , [1, query_ent_start, query_ent_end]],
                                            label])
            return all_samples
        preprocess('ReCoRD', args, 2, process_func, nspans=2)
    if 'RTE' in tasks:
        def process_func(example):
            try:
                label = str(example['label'])
            except:
                label = -1
            return [ # one sample
                    [[example["premise"], example["hypothesis"]] , label]
                    ]
        preprocess('RTE', args, 2, process_func)
    if 'WiC' in tasks:
        def process_func(example):
            if (example['idx'] in [1696,2300,2970,4085,4718] ) and example['word']=='do':
                example["sentence2"] = example["sentence2"].replace('doesn\'t','does not')
                checkalignment(example["sentence2"], example['start2'],example['end2'])
            elif (example['idx'] in [780] ) and example['word']=='do':
                example["sentence2"] = example["sentence2"].replace('Don\'t','do not')
                checkalignment(example["sentence2"], example['start2'],example['end2'])
            elif (example['idx'] in [788] ) and example['word']=='have':
                example["sentence2"] = example["sentence2"].replace('\'ve',' have')
                example['start2'] = example['start2']+1
                example['end2'] = example['end2']+2
                checkalignment(example["sentence2"], example['start2'],example['end2'])

            try:
                label = str(example['label'])
            except:
                label = -1
            try:
                checkalignment(example["sentence1"], example['start1'],example['end1'])
            except:
                import pdb; pdb.set_trace()

            try:
                checkalignment(example["sentence2"], example['start2'],example['end2'])
            except:
                import pdb; pdb.set_trace()
            return [ # one sample, with inverse
                    [[example["sentence1"], example["sentence2"]] ,
                        [[0, example['start1'],example['end1']] , [1, example['start2'],example['end2']]]
                        , label],
                    [[example["sentence2"], example["sentence1"]] ,
                        [[0, example['start2'],example['end2']] , [1, example['start1'],example['end1']]]
                        , label]
                    ]
        preprocess('WiC', args, 2, process_func, nspans=2)
    # if 'WSC' in tasks:
    #     pass
    # os.remove("preprocess.py")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
