import json
import os
import csv
import numpy as np
from unicodedata import normalize
from collections import defaultdict
from random import shuffle

"""
wget https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
"""

def main():
    # path to wiki-dump files
    path_to_data = r"wiki-pages/"
    # collect the names of all wiki files
    json_files = [fname for fname in os.listdir(path_to_data) if fname.startswith('wiki-') and fname.endswith('.jsonl')]

    def choice_from_list(some_list, sample_size):
        """
        Input: a list and desired sample size
        Returns: a random sample of the desired size, the same list with this sample removed
        """
        res = np.random.choice(some_list, sample_size, replace = False)
        new_list = [x for x in some_list if x not in res]
        return res, new_list

    # compile wiki files into one massive dictionary
    wiki_data = {}
    for i,f in enumerate(json_files):
        with open(os.path.join(path_to_data, f)) as jreader:
            for itm in jreader:
                j = json.loads(itm)
                wiki_data[normalize('NFC', j['id'])] = j['lines'].split('\n')
            print("Parsing {} of {} data chunks. Total entries: {}".format(i+1, len(json_files), len(wiki_data)))
        break


    # create csv to write to (split into train, test, eval sets)
    with open('nli_fever.jsonl', 'w+') as fout:
        # collect evidence ids and pair to wiki_data based on sentence index
        for infile in ['fever/train.jsonl']:
            with open(infile) as jreader:
                for itm in jreader:
                    j = json.loads(itm)
                    id = str(j['id'])
                    newitem = {'id':id}

                    # change 'verifiable' and 'label' into integers for easier manipulation
                    if j['verifiable'] == 'VERIFIABLE':
                        newitem['verifiable'] = 1
                    else:
                        newitem['verifiable'] = 0
                    if j['label'] == 'SUPPORTS':
                        newitem['label'] = 'entailment'
                    elif j['label'] == 'REFUTES':
                        newitem['label'] = 'contradiction'
                    else:
                        newitem['label'] = 'neutral'
                    newitem['hypothesis'] = j['claim']
                    newitem['premise'] = []
                    # add all evidence pieces
                    for e in j['evidence']:
                        for evidence in e:
                            anno_id = evidence[0]
                            evidence_id = evidence[1]
                            article_name = evidence[2]
                            sentence_id = evidence[3]
                            if sentence_id is not None:
                                try:
                                    article_name = normalize('NFC', article_name)
                                    current_sentence = wiki_data[article_name][sentence_id].split('\t')[1]
                                    if current_sentence not in newitem['premise']:
                                        newitem['premise'].append(current_sentence)
                                except KeyError as e:
                                    print(article_name, ' is not in available evidence.')
                                    pass
                    if len(newitem['premise']) > 0:
                        newitem['premise'] = ''.join(newitem['premise']).replace('\n',' ')
                        json.dump(newitem, fout)
                        fout.write('\n')
                    else:
                        print(id, ' has no evidence.')                        


if __name__ == '__main__':
    main()
