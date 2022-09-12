import os
import re
import torch
import subprocess
from transformers import *
import random

pos_converter = {'NOUN':'n', 'PROPN':'n', 'VERB':'v', 'AUX':'v', 'ADJ':'a', 'ADV':'r'}

def generate_key(lemma, pos):
    key = '{}#{}'.format(lemma.lower(), pos)
    return key

def load_pretrained_model(name):
    if name == 'bert-large-multilingual-uncased':
        model = BertModel.from_pretrained('bert-large-multilingual-uncased')
        hdim = 1024
    else: #bert base
        model = BertModel.from_pretrained('bert-base-multilingual-uncased')
        hdim = 768
    return model, hdim

def load_tokenizer(name):
    if name == 'bert-large-multilingual-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-large-multilingual-uncased')
    else: #bert base
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    return tokenizer

def load_bn_senses(path):
    wn_senses = {}
    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip().split('\t')
            senses = line[1:]
            key = line[0]
            wn_senses[key] = senses
    return wn_senses

def get_label_space(data):
    #get set of labels from dataset
    labels = set()
    
    for sent in data:
        for _, _, _, _, label in sent:
            if label != -1:
                labels.add(label)

    labels = list(labels)
    labels.sort()
    labels.append('n/a')

    label_map = {}
    for sent in data:
        for _, lemma, pos, _, label in sent:
            if label != -1:
                key = generate_key(lemma, pos)
                label_idx = labels.index(label)
                if key not in label_map: label_map[key] = set()
                label_map[key].add(label_idx)

    return labels, label_map

def process_encoder_outputs(output, mask, as_tensor=False):
    combined_outputs = []
    position = -1
    avg_arr = []
    for idx, rep in zip(mask, torch.split(output, 1, dim=0)):
        #ignore unlabeled words
        if idx == -1: continue
        #average representations for units in same example
        elif position < idx: 
            position=idx
            if len(avg_arr) > 0: combined_outputs.append(torch.mean(torch.stack(avg_arr, dim=-1), dim=-1))
            avg_arr = [rep]
        else:
            assert position == idx 
            avg_arr.append(rep)
    #get last example from avg_arr
    if len(avg_arr) > 0: combined_outputs.append(torch.mean(torch.stack(avg_arr, dim=-1), dim=-1))
    if as_tensor: return torch.cat(combined_outputs, dim=0)
    else: return combined_outputs

#run WSD Evaluation Framework scorer within python
def evaluate_output(scorer_path, gold_filepath, out_filepath):
    eval_cmd = ['java','-cp', scorer_path, 'Scorer', gold_filepath, out_filepath]
    print(eval_cmd)
    #output = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE ).communicate()[0]
    output = subprocess.check_output(eval_cmd)
    output = [x.decode("utf-8") for x in output.splitlines()]
    p,r,f1 =  [float(output[i].split('=')[-1].strip()[:-1]) for i in range(3)]
    return p, r, f1


def load_data(filename):

    pre_sentence = ''
    #load train examples + annotate sense instances with gold labels
    ret_sentences = []
    s = []

    indices = []
    insts = []
    lemmas = []
    poses = []
    sense_labels = []
    with open(filename, 'r', encoding="utf8") as f:
        f.readline()
        for num, line in enumerate(f):
            info = line.strip().split('\t')
            sentence, start_idx, end_idx, inst, lemma, pos, sense_label = info
            if sentence != pre_sentence and pre_sentence != '':
                words = pre_sentence.strip().split(' ')
                all_length = len(words)
                ind_for_sense = 0
                i = 0
                while i < all_length:
                    if ind_for_sense < len(indices) and i < indices[ind_for_sense][0]:
                        # information is not important for unlabeled word
                        s.append((words[i], words[i], 'DET', -1, -1))
                        i += 1
                    elif ind_for_sense >= len(indices):
                        s.append((words[i], words[i], 'DET', -1, -1))
                        i += 1
                    elif i == indices[ind_for_sense][0]:
                        s.append((' '.join(words[indices[ind_for_sense][0]:indices[ind_for_sense][1]]), 
                            lemmas[ind_for_sense], poses[ind_for_sense], insts[ind_for_sense], sense_labels[ind_for_sense]))
                        i = indices[ind_for_sense][1]
                        ind_for_sense += 1
                ret_sentences.append(s)
                indices = []
                insts = []
                lemmas = []
                poses = []
                sense_labels = []
                s = []

            pre_sentence = sentence
            indices.append([int(start_idx), int(end_idx)])
            insts.append(inst)
            lemmas.append(lemma)
            poses.append(pos)
            sense_labels.append(sense_label)

        # last sentence
        words = pre_sentence.strip().split(' ')
        all_length = len(words)
        
        ind_for_sense = 0
        i = 0
        while i < all_length:
            if ind_for_sense < len(indices) and i < indices[ind_for_sense][0]:
                # information is not important for unlabeled word
                s.append((words[i], words[i], 'DET', -1, -1))
                i += 1
            elif ind_for_sense >= len(indices):
                s.append((words[i], words[i], 'DET', -1, -1))
                i += 1
            elif i == indices[ind_for_sense][0]:
                s.append((' '.join(words[indices[ind_for_sense][0]:indices[ind_for_sense][1]]), 
                    lemmas[ind_for_sense], poses[ind_for_sense], insts[ind_for_sense], sense_labels[ind_for_sense]))
                i = indices[ind_for_sense][1]
                ind_for_sense += 1

    return ret_sentences

#normalize ids list, masks to whatever the passed in length is
def normalize_length(ids, attn_mask, o_mask, max_len, pad_id):
    if max_len == -1:
        return ids, attn_mask, o_mask
    else:
        if len(ids) < max_len:
            while len(ids) < max_len:
                ids.append(torch.tensor([[pad_id]]))
                attn_mask.append(0)
                o_mask.append(-1)
        else:
            ids = ids[:max_len-1]+[ids[-1]]
            attn_mask = attn_mask[:max_len]
            o_mask = o_mask[:max_len]

        assert len(ids) == max_len
        assert len(attn_mask) == max_len
        assert len(o_mask) == max_len

        return ids, attn_mask, o_mask

