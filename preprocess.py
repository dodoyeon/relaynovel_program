import os
import csv
import glob
import re
from pathlib import Path

from pykospacing import Spacing
# import kss
from kiwipiepy import Kiwi
from soynlp.normalizer import *
from transformers import PreTrainedTokenizerFast

kiwi = Kiwi()
spacing = Spacing()

# remove special char -> kospacing -> kss -> punctuation
# not use phrase start token or novel start token (and end token)

def korean_lib(text):
    split_list = kiwi.split_into_sents(text)
    new_list = []
    for s in split_list:
        sent = spacing(s.text)
        sent = repeat_normalize(sent, num_repeats=2)
        new_list.append(sent)
    return new_list

def add_spetoken(text, msl):
    with open(dir_path + novel, 'r') as f:
        lreader = csv.reader(f)
        new_list = []
        temp = ''
        for line in lreader:
            line = ''.join(line)
            if temp != '':
                line = temp +' '+ line
                temp = ''
            if len(line) <= 4:
                temp = line
            else:
                temp = ''
                # line = '<s>' + line + '</s>' # not add spetoken here
                new_list.append(line)
    with open(dir_path + 'preprocess_final.csv', 'a', newline='') as label: # Open labeling file
        lwriter = csv.writer(label)
        for s in new_list:
            lwriter.writerow([s])
    print('done')

def csv_to_text(file_name, msl): # for Trainer()
    zipped_data = []
    phrase = []
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                            bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                                            pad_token='<pad>', mask_token='<mask>')
    
    with open(file_name, 'r') as f:
        lreader = csv.reader(f)
        temp = ''
        for sent in lreader:
            toks = tokenizer.tokenize(sent)
            if len(phrase) + len(toks) > msl - 2:
                # phrase = ['<s>'] + phrase + ['</s>']
                zipped_data.append(phrase)
                phrase = temp + sent
                temp = ''
            else:
                phrase += sent
                temp = sent
    
    with open('preprocess_final.txt', 'a', encoding='utf-8') as f:
        for sent in zipped_data:
            f.write(sent+'\n')

def preprocess_v2(dir_path,file_path):
    # special_chars = set()
    # japanese_pattern = re.compile("[ぁ-ゔ]|[ァ-ヴー]|[々〆〤]")
    # chinese_pattern = re.compile("[一-鿕]|[㐀-䶵]|[一-龥]|[豈-龎]") # 
    # korean_pattern = re.compile("[가-힣ㄱ-ㅎㅏ-ㅣ]")

    with open(dir_path+file_path, 'r', encoding='utf-8') as f:
        # for sent in reader:
        #     sent[0] = re.sub(chinese_pattern, "", sent[0])
        #     sent[0] = re.sub(japanese_pattern, "", sent[0])
        #     special_chars.update(re.findall("[^가-힣ㄱ-ㅎㅏ-ㅣ,.?! \"\']", sent[0]))
        text = f.read()
    text = text.replace('…', '...')
    text = re.sub("[^가-힣 ,.?!]", "", text)
    mapping = {' .':'', '   ':' ', '  ':' ', '??': '?'}
    for p in mapping:
        text = text.replace(p, mapping[p])
    preprocessed_list = korean_lib(text)
    with open('preprocess.csv', 'a', newline='') as f:
        lwriter = csv.writer(f)
        for sent in preprocessed_list:
            lwriter.writerow([sent])
    print('done')

dir_path = 'preprocessed/'# 'preprocessed/prepre/'
novel_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
# novel_files = novel_files[35:]
for novel in novel_files:
    preprocess_v2(dir_path, novel)