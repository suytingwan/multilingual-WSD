import numpy as np
from collections import defaultdict


def read_bn_gloss(glossfiles, glossout):
    '''
    glossfiles: the request result from BabelNet API
    glossout:
         bn_word    source    definition
    ''' 
    dicts = {}
    for glossfile in glossfiles:
        sources = []
        glosses = []
        prebn = ''
        for line in open(glossfile):
            info = line.strip().split('\t')
            if info[0] in dicts.keys():
                pass
            elif info[0] != prebn and prebn != '':
                if len(sources) > 0:
                    if 'WN' in sources:
                        ind = sources.index('WN')
                        dicts[prebn] = glosses[ind]
                    else:
                        dicts[prebn] = glosses[0]
                sources = []
                glosses = []
                prebn = info[0]
                sources.append(info[2])
                glosses.append(info[3])
            else:
                sources.append(info[2])
                glosses.append(info[3])
                prebn = info[0]

        if len(sources) > 0:
            if 'WN' in sources:
                ind = sources.index('WN')
                dicts[info[0]] = glosses[ind]
            else:
                dicts[info[0]] = glosses[0]

    fw = open(glossout, 'w')
    for key in dicts.keys():
        fw.write('{}\t{}\n'.format(key, dicts[key]))
    fw.close()


if __name__ == "__main__":

    # split in ['de', 'fr', 'it', 'es']
    split = 'de'
    infile = './mulan-{}/{}_bn_gloss.txt'.format(split, split)
    outfile = './mulan-{}/{}_bn_wn_gloss.txt'.format(split, split)
    read_bn_gloss(infile, outfile)
