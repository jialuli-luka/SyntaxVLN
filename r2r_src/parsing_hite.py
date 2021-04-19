import stanza
import json

nlp = stanza.Pipeline("hi", tokenize_no_ssplit=True)

env = ['train', 'val_seen', 'val_unseen']

for name in env:
    new_data = []
    with open('tasks/RXR/rxr-data/rxr_%s_guide.jsonl' % name) as f:
        for line in f:
            new_data.append(json.loads(line))
    f.close()

    print("total:",len(new_data))
    for i, item in enumerate(new_data):
        print("i",i)
        if item['language'] != 'hi-IN':
            continue
        new_data[i]['parse_tree'] = []
        new_data[i]['parse_tree_origin'] = []
        new_data[i]['token'] = []
        instr = item['instruction']
        doc = nlp(instr)
        parse_dict = dict()
        for sent in doc.sentences:
            for word in sent.words:
                parse_dict[str(word.id)] = {"id":word.id, "word":word.text, "lemma":word.lemma, "upos":word.upos, "xpos":word.xpos, "feats":word.feats, "head": word.head, "deprel": word.deprel, "misc":word.misc}
        new_data[i]['parse_tree_origin'] = [parse_dict]
        new_data[i]['instructions'] = [item['instruction']]

    with open('tasks/RXR/rxr-data/rxr_%s_guide_dep_hi.jsonl' % name, 'w') as f:
        json.dump(new_data, f)
    f.close()

nlp = stanza.Pipeline("te", tokenize_no_ssplit=True)

env = ['train', 'val_seen', 'val_unseen']

for name in env:
    new_data = []
    with open('tasks/RXR/rxr-data/rxr_%s_guide.jsonl' % name) as f:
        for line in f:
            new_data.append(json.loads(line))
    f.close()

    print("total:",len(new_data))
    for i, item in enumerate(new_data):
        print("i",i)
        if item['language'] != 'te-IN':
            continue
        new_data[i]['parse_tree'] = []
        new_data[i]['parse_tree_origin'] = []
        new_data[i]['token'] = []
        instr = item['instruction']
        doc = nlp(instr)
        parse_dict = dict()
        for sent in doc.sentences:
            for word in sent.words:
                parse_dict[str(word.id)] = {"id":word.id, "word":word.text, "lemma":word.lemma, "upos":word.upos, "xpos":word.xpos, "feats":word.feats, "head": word.head, "deprel": word.deprel, "misc":word.misc}
        new_data[i]['parse_tree_origin'] = [parse_dict]
        new_data[i]['instructions'] = [item['instruction']]

    with open('tasks/RXR/rxr-data/rxr_%s_guide_dep_te.jsonl' % name, 'w') as f:
        json.dump(new_data, f)
    f.close()
