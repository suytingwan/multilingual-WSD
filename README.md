This is the code repo for COLING2022: Multilingual Word Sense Disambiguation with Unified Sense Representation.

## Envs
   python 3.7
   transformer 2.6

## BabelNet, Inventory and Evaluation benchmark

   [BabelNet](https://babelnet.org) is a multilingual semantic lexicon which contains inventories for multiple languages. Inventories and multilingual WSD benchmark can be found at [https://github.com/SapienzaNLP/mwsd-datasets](https://github.com/SapienzaNLP/mwsd-datasets). For MWSD evaluation, we use the wn split. After handling the inventories and evaluation dataset, the data are placed under `./data`. The xml and gold key file in evaluation are further converted into csv format by:

   ```
   cd ./preprocess
   python convert_xml_csv.py
   ```
   A small percentage (20%) is sampled from semeval-2013 evaluation dataset for model selection and placed under `./preprocess`.

## Machine Translation and Fastalign
   First acquire the transalted corpora for languages DE, FR, IT and ES:
   ```
   cd ./translation
   python translate.py
   ```
   Then, use [fastalign tool](https://github.com/clab/fast_align) to get the alignment between orginal semcor and translated corpora. This step will result alignment file under same folder.

   Finally, generate training corpora for multiple languages by:
   ```
   python mapping_synset.py
   ```


## Online lexical knowledge gloss request
   For experiments conducted on [MuLaN dataset](https://github.com/SapienzaNLP/mwsd-datasets), the original data are filtered by inventories (WordNet split). 

   MuLaN contains some Babel synsets which do not exist in WordNet. For feasible experiments(collecting glosses according to inventories), we filter out the training instances labeled with words which are not shown in wn-split inventory generated from BabelNet.
   ```
   cd mulan
   python filter_mulan.py  
   ``` 

   For senses covered by annotated words but not shown in WordNet, we request BabelNet API to get gloss knowledge for the senses.

   ```
   python request.py
   python read_bn_gloss.py 
   ```

## Experiments 
   The difference between translated data and MuLaN is the different training data, inventories, and glosses. Since no validation data provided, a small percentage of test data from semevel13 is used as evaluation. For easy experiments, our code and preprocessed data can be download at [code and data](https://drive.google.com/drive/folders/1G79S-t2Li867bvDPjh2zQZMhgJa6CkKK?usp=sharing).

### Experiments with translated data

   ```
   CUDA_VISIBLE_DEVICES=3,0 python biencoder_mwsd.py --ckpt ./ckpts/trans_de \
        --data-path ./Evaluation \
        --valid-path ./preprocess \
        --train-path ./translation/train_corpora \
        --gloss-path ./mulan/mulan-de/gloss_new_combine.txt \
        --inventory-path ./mulan/mulan-de/de_inventory_filter_new_combine.txt \
        --grad-bsz 30 \
        --split semeval2013-de \
        --multigpu \
        --valid-small
   ```


### Experiments with MuLaN

   ```
   CUDA_VISIBLE_DEVICES=0,1 python biencoder_mwsd.py --ckpt ./ckpts/mulan_de \
        --data-path ./Evaluation \
        --valid-path ./preprocess \
        --train-path ./mulan/mulan-de/transfer_filter_new_check.csv \
        --gloss-path ./mulan/mulan-de/gloss_new_combine.txt \
        --inventory-path ./mulan/mulan-de/de_inventory_filter_new_combine.txt \
        --grad-bsz 30 \
        --split semeval2013-de \
        --multigpu \
        --valid-small
   ```

### Evaluation

   ```
   CUDA_VISIBLE_DEVICES=0,1 python biencoder_mwsd.py --ckpt ./ckpts/trans_de \
        --data-path ./Evaluation \
        --valid-path ./preprocess \
        --train-path ./translation/train_corpora \
        --gloss-path ./mulan/mulan-de/gloss_new_combine.txt \
        --inventory-path ./mulan/mulan-de/de_inventory_filter_new_combine.txt \
        --grad-bsz 30 \
        --split semeval2013-de \
        --multigpu \
        --eval
   ```


