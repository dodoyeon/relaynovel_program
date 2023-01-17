import os
import csv
import glob
import re
from pathlib import Path

# from pykospacing import Spacing
# import kss
# from kiwipiepy import Kiwi
# from soynlp.normalizer import *
# from transformers import PreTrainedTokenizerFast

# kiwi = Kiwi()
# spacing = Spacing()

def space_normalize(dir_path, novel, save=True):
    f = open(dir_path+novel, 'r', encoding='utf-8')
    text = f.read()
    f.close()
    # text = text.replace("...", ".")
    # text = text.replace(" .", ".")
    # text = text.replace("..", ".")
    # text = text.replace("  ", " ")
    # text = text.replace('  ',' ')
    # text = text.replace("  ", " ")
    # text = text.replace("??", "?")
    # text = text.replace("「", "")
    # text = text.replace("」", "")
    # text = text.replace("中", "")

    # text = text.replace("'", "")
    # text = text.replace('"','')
    # text = text.replace("♡", "")
    # text = text.replace(" .", "")
    # text = text.replace(" ,", "")
    # text = text.replace(".,", "")

    unused_chars = ["'", "\"", "♡", " .", " ,", ".,"]
    for c in unused_chars:
        text = text.replace(c, "")

    # f = open(dir_path+novel, 'w', encoding='utf-8')
    # f.write(text)
    # f.close()
    if save:
        with open(dir_path+novel, 'w', encoding='utf-8') as f:
            f.write(text)

def soy_normalize(dir_path, novel):
    f = open(dir_path+novel, 'r', encoding='utf-8')
    text = f.read()
    f.close()
    text = emoticon_normalize(text, num_repeats=2)
    text = repeat_normalize(text, num_repeats=2)
    f.close()
    f = open(dir_path+novel, 'w', encoding='utf-8')
    f.write(text)
    f.close()

# remove special char -> kospacing -> kss -> punctuation
# not use phrase start token or novel start token (and end token)

def kiwi_use(dir_path, novel):
    novel = open(dir_path + novel, 'r', encoding='utf-8')
    text = novel.read()
    novel.close()
    split_list = kiwi.split_into_sents(text)
    for s in split_list:
        text = spacing(s.text)
        with open('preprocess.csv', 'a', newline='') as label: # Open labeling file
            lwriter = csv.writer(label)
            lwriter.writerow([text])

def add_spetoken(dir_path, novel):
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

def delete_wrong_spetoken(novel): # NOT WORKING!!!
    new_list = []
    with open(novel, 'r') as f:
        lreader = csv.reader(f)
        for line in lreader:
            line = ''.join(line)
            # line.replace('<s>', '')
            # line.replace('</s>', '')
            new_list.append(line)
    with open('preprocess_fin.csv', 'a', newline='') as label: # Open labeling file
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
                phrase = ['<s>'] + phrase + ['</s>']
                zipped_data.append(phrase)
                phrase = temp + sent
                temp = ''
            else:
                phrase += sent
                temp = sent
    
    with open('preprocess_final.txt', 'a', encoding='utf-8') as f:
        for sent in zipped_data:
            f.write(sent+'\n')

def remove_special_chars(file_path):
    special_chars = set()
    file = Path('data/preprocessed/preprocess_final.csv')
    japanese_pattern = re.compile("[ぁ-ゔ]|[ァ-ヴー]|[々〆〤]")
    chinese_pattern = re.compile("[一-鿕]|[㐀-䶵]|[一-龥]|[豈-龎]") # 
    korean_pattern = re.compile("[가-힣ㄱ-ㅎㅏ-ㅣ]")

    with file.open('r', encoding='utf-8') as f:
        reader = csv.reader(f, )
        for sent in reader:
            sent[0] = re.sub(chinese_pattern, "", sent[0])
            sent[0] = re.sub(japanese_pattern, "", sent[0])
            special_chars.update(re.findall("[^가-힣ㄱ-ㅎㅏ-ㅣ,.?! \"\']", sent[0]))
    print(special_chars)

# dir_path = 'preprocessed/'# 'preprocessed/prepre/'
# novel_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))][2:]
# novel_files = novel_files[35:]
# for novel in novel_files:
#     space_normalize(dir_path, novel)
#     soy_normalize(dir_path, novel)
#     kospacing_use(dir_path, novel)
#     kiwi_use(dir_path, novel)

# add_spetoken('', 'preprocess.csv')
# delete_wrong_spetoken('preprocess_final.csv')
# csv_to_text('preprocess_final.csv', 256)