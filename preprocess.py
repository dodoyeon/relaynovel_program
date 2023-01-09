import os
import csv
from pykospacing import Spacing
# import kss
from kiwipiepy import Kiwi
from soynlp.normalizer import *

kiwi = Kiwi()
spacing = Spacing()

def space_normalize(dir_path, novel):
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
    text = text.replace("'", "")
    text = text.replace('"','')
    text = text.replace("♡", "")
    text = text.replace(" .", "")
    text = text.replace(" ,", "")
    text = text.replace(".,", "")
    f = open(dir_path+novel, 'w', encoding='utf-8')
    f.write(text)
    f.close()

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

def kospacing_use(dir_path, novel):
    f = open(dir_path + novel, 'r', encoding='utf-8')
    text = f.read()
    f.close()
    text = text.replace("♡", "")
    text = text.replace(" .", "")
    f = open('prespaced/'+novel, 'w', encoding='utf-8')
    f.write(text)
    f.close()

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
            line.replace('<s>', '')
            line.replace('</s>', '')
            new_list.append(line)
    with open('preprocess_fin.csv', 'a', newline='') as label: # Open labeling file
        lwriter = csv.writer(label)
        for s in new_list:
            lwriter.writerow([s])
    print('done')


# dir_path = 'preprocessed/'# 'preprocessed/prepre/'
# novel_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))][2:]
# novel_files = novel_files[35:]
# for novel in novel_files:
#     space_normalize(dir_path, novel)
#     soy_normalize(dir_path, novel)
#     kospacing_use(dir_path, novel)
#     kiwi_use(dir_path, novel)

# add_spetoken('', 'preprocess.csv')
delete_wrong_spetoken('preprocess_final.csv')