import os
import re
import subprocess
import random
from collections import defaultdict
import xml.etree.ElementTree as ET
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

def load_data(datapath, name):
    text_path = os.path.join(datapath, '{}.data.xml'.format(name))
    gold_path = os.path.join(datapath, '{}.gold.key.txt'.format(name))

    #load train examples + annotate sense instances with gold labels
    tree = ET.ElementTree(file=text_path)
    root = tree.getroot()

    sentences = []
    poss = []
    targets = []
    targets_index_start = []
    targets_index_end = []
    lemmas = []

    for doc in root:
        for sent in doc:
            sentence = []
            pos = []
            target = []
            target_index_start = []
            target_index_end = []
            lemma = []
            for token in sent:
                assert token.tag == 'wf' or token.tag == 'instance'
                if token.tag == 'wf':
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append('X')
                        lemma.append(token.attrib['lemma'])
                if token.tag == 'instance':
                    target_start = len(sentence)
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append(token.attrib['id'])
                        lemma.append(token.attrib['lemma'])
                    target_end = len(sentence)
                    assert ' '.join(sentence[target_start:target_end]) == token.text
                    target_index_start.append(target_start)
                    target_index_end.append(target_end)
            sentences.append(sentence)
            poss.append(pos)
            targets.append(target)
            targets_index_start.append(target_index_start)
            targets_index_end.append(target_index_end)
            lemmas.append(lemma)

    gold_keys = {}
    for line in  open(gold_path, 'r', encoding="utf8"):
        key = line.strip().split(' ')
        gold_keys[key[0]] = key[1]

    outfile = name + '.csv'
    with open(outfile, 'w', encoding="utf8") as g:
        g.write('sentence\ttarget_index_start\ttarget_index_end\ttarget_id\ttarget_lemma\ttarget_pos\tsense_key\n')
        num = 0
        for i in range(len(sentences)):
            for j in range(len(targets_index_start[i])):
                sentence = ' '.join(sentences[i])
                target_start = targets_index_start[i][j]
                target_end = targets_index_end[i][j]
                target_id = targets[i][target_start]
                target_lemma = lemmas[i][target_start]
                target_pos = poss[i][target_start]
                #sense_key = gold_keys[num]
                if target_id not in set(gold_keys.keys()):
                    num += 1
                    continue
                sense_key = gold_keys[target_id]
                g.write('\t'.join((sentence, str(target_start), str(target_end), target_id, target_lemma, target_pos, sense_key)))
                g.write('\n')
    #which means some words are not considered as test senses
    print('missing gold keys: ', num)

if __name__ == "__main__":

    # split in ['semeval2013-de', 'semeval2013-es', 'semeval2013-fr', 'semeval2013-it', 'semeval2015-es', 'semeval2013-it']
    #split = 'semeval2013-de'
    #data_path = '../Evaluation/multilingual_wsd_wn_v1.0/{}'.format(split)
    #load_data(data_path, split) 

    split = 'semcor'
    data_path = '../Evaluation/Training_Corpora/SemCor/'
    load_data(data_path, split)
