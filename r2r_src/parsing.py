import json
from nltk.parse.corenlp import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser
import re
import string
from nltk.tree import *


def split_sentence(sentence):
    toks = []
    for word in [s.strip().lower() for s in re.compile(r'(\W+)').split(sentence.strip()) if
                 len(s.strip()) > 0]:
        # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
        if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
            toks += list(word)
        else:
            toks.append(word)
    return toks


if __name__ == '__main__':

    env = ['train', 'val_seen', 'val_unseen']

    parser = CoreNLPDependencyParser(url='http://localhost:9000')

    for split in env:
        with open('tasks/R2R/data/R2R_%s.json' % split) as f:
            new_data = json.load(f)
        f.close()

        print("total:",len(new_data))
        for i, item in enumerate(new_data):
            print("i",i)
            new_data[i]['parse_tree'] = []
            new_data[i]['parse_tree_origin'] = []
            for j, instr in enumerate(item['instructions']):
                splited_instr = split_sentence(instr)
                parse = next(parser.parse(splited_instr))
                new_data[i]['parse_tree_origin'].append(parse.nodes)        #dep
                new_data[i]['parse_tree'].append(str(parse.tree()))         #dep

        with open('tasks/R2R/data/R2R_%s_depparse.json' % split, 'w') as f:
            json.dump(new_data, f)
        f.close()

    for split in env:
        new_data = []
        with open('tasks/RXR/rxr-data/rxr_%s_guide.jsonl' % split) as f:
            for line in f:
                new_data.append(json.loads(line))
        f.close()
        for i, item in enumerate(new_data):
            print("i", i)
            if item['language'] != 'en-US' and item['language'] != 'en-IN':
                continue
            new_data[i]['parse_tree'] = []
            new_data[i]['parse_tree_origin'] = []
            instr = item['instruction']
            splited_instr = split_sentence(instr)
            parse = next(parser.parse(splited_instr))
            new_data[i]['parse_tree_origin'] = [parse.nodes]          #dep
            new_data[i]['parse_tree'] = [str(parse.tree())]       #dep
            new_data[i]['instructions'] = [item['instruction']]

        with open('tasks/RXR/rxr-data/rxr_val_seen_guide_dep.jsonl', 'w') as f:
            json.dump(new_data, f)
        f.close()