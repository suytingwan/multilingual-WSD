import numpy as np
from collections import defaultdict


def readword(filein, fileout):
    '''
    filein: the csv file extracted from MuLan input
    '''

    dicts = defaultdict(list)
    fread = open(filein)
    csv_lines = []
    for i, line in enumerate(fread):
        if i == 0:
            csv_lines.append(line)
            continue

        csv_lines.append(line)
        info = line.strip().split('\t')
        key = info[4].lower() + '#' + info[5]
        if info[6] in dicts[key]:
            pass
        else:
            dicts[key].append(info[6])

    print(len(dicts.keys()))

    fw = open(fileout, 'w')
    for key in dicts.keys():
        new_line = '{}\t{}\n'.format(key, '\t'.join(dicts[key]))
        fw.write(new_line)
    fw.close()
    return csv_lines


def filter_wn_words(inmulan, inventoryfile, wordfile, filterins, filtermulan):
    '''
    wordfile: the extracted word+pos from training inventory
    inventoryfile: wordnet inventory in foreign language
    '''
    dicts = {}
    fread = open(inventoryfile)
    for line in fread:
        info = line.strip().split('\t')
        dicts[info[0]] = line
    fread.close()

    fread2 = open(wordfile)
    fw = open(filterins, 'w')
    for line in fread2:
        info = line.strip().split('\t')
        if info[0] in dicts.keys():
            fw.write(dicts[info[0]])
    fw.close()
    fread2.close()

    fw2 = open(filtermulan, 'w')
    fw2.write(inmulan[0])
    for mulan_ins in inmulan[1:]:
        info = mulan_ins.strip().split('\t')
        key = info[4].lower() + '#' + info[5]
        if key in dicts.keys():
            fw2.write(mulan_ins)
    fw2.close()


if __name__ == "__main__":

    #split in ['de', 'es', 'it', 'fr']
    split = 'de'

    csv_lines = readword('./mulan-{}/transfer.csv'.format(split), './mulan-{}/{}_inventory_train_raw.txt'.format(split, split))
    filter_wn_words(csv_lines, '../Evaluation/multilingual_wsd_wn_v1.0/inventories/{}/inventory.{}.withgold.txt'.format(split, split), \
        './mulan-{}/{}_inventory_train_raw.txt'.format(split, split), \
        './mulan-{}/{}_inventory_train_filter.txt'.format(split, split), \
        './mulan-{}/transfer_filter.csv'.format(split))

