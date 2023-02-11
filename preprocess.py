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
    preprocessed_list = korean_lib(text) # kiwi + kss
    with open('preprocess_fin.csv', 'a', newline='') as f:
        lwriter = csv.writer(f)
        for sent in preprocessed_list:
            lwriter.writerow([sent])
    print('done')

def csv_to_phrasetxt(file_name, msl): # for Trainer()
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                            bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                                            pad_token='<pad>', mask_token='<mask>')
    
    # tokenizer.add_special_tokens({'additional_special_token':['<sep>']})
    phrase_list = []
    templ = []
    with open(file_name, 'r') as f:
        lreader = csv.reader(f)
        phrase = ''
        len_p = 0
        for sent in lreader:
            templ.append(sent)
            sent = ''.join(sent)
            toks = tokenizer.tokenize(sent)
            len_t = len(toks)+1
            if len_p + len_t <= msl - 2:
                phrase += (sent + ' ')
                len_p += len_t
            else:
                if len_p > msl-2:
                    phrase = toks[:(msl-3)]
                phrase_list.append(phrase)
                phrase = (sent +' ')
                len_p = len_t

    with open('phrase_final.txt', 'a', encoding='utf-8') as f:
        for sent in phrase_list:
            f.write(sent+'\n')

def csv_to_senttxt(file_name, msl): # for Trainer()
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                            bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                                            pad_token='<pad>', mask_token='<mask>')
    sent_list = []
    num = 0 
    with open(file_name, 'r') as f:
        lreader = csv.reader(f)
        for sent in lreader:
            sent = ''.join(sent)
            toks = tokenizer.tokenize(sent)
            if len(toks) > msl - 2:
                num += 1
            else:
                sent_list.append(sent)
    with open('sent_final.txt', 'a', encoding='utf-8') as f:
        for sent in sent_list:
            f.write(sent+'\n')


# dir_path = 'prepre/'
# novel_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
# novel_files = novel_files[5:]
# for novel in novel_files:
#     preprocess_v2(dir_path, novel)
csv_to_senttxt('preprocess_fin.csv', 256)