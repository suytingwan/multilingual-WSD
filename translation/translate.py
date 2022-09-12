from googletrans import Translator
import time

translator = Translator()
translator.raise_Exception = True

splits = ['de', 'fr', 'es', 'it']
for split in splits[0:]:
    fw = open('./trans_copora/{}_trans_semcor.txt'.format(split), 'a+')
    fread = open('../preprocess/semcor.csv')
    fread.readline()
    for line in fread:
        info = line.strip().split('\t')[0]
        try:
            result = translator.translate(info, src='en', dest=split)
            fw.write(result.origin + '#!!!#' + result.text + '\n')
            time.sleep(0.5)
        except:
            print('error in translating one line')
    fw.close()
