from utils import *
import json
from collections import Counter, defaultdict

base_vocab = ['<PAD>', '<UNK>', '<EOS>']

def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    # t = Tokenizer()
    with open('../RXR/rxr-data/rxr_%s_guide_dep.jsonl' % 'train') as f:
        data = json.load(f)
    for item in data:
        for instr in item['instructions']:
            count.update(t.split_sentence(instr))
    vocab = list(start_vocab)
    for word,num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab