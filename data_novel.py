import os
import csv
from torch.utils.data import Dataset
import torch

class NovelDataSet(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.load_data()

    def load_data(self):
        with open(self.file_path, 'r') as f:
            self.data = list(csv.reader(f))

    def __len__(self):
        # data_list = os.listdir(self.file_path)
        # if hasattr(self, "_length"): # if _length in self
        #     return self._length

        # with open(self.file_path, 'r') as f:
        #     length = len(list(csv.reader(f)))
        #     self._length = length
        if not hasattr(self, 'data'):
            return 0
        return len(self.data)

    def __getitem__(self, index):
        # file_name = 'novel_sentence'+str(index)+'.txt'
        # text_path = os.path.join(self.file_path, file_name)
        # with open(self.file_path, 'r', encoding = 'utf-8') as file:
            # line = file.read()
            # lreader = csv.reader(file)
            # line = list(lreader)[index] # ???????
        line = self.data[index]
        
        # transform
        line = self.tokenizer.tokenize(''.join(line))
        if len(line) > self.max_length - 2:
            line = line[:self.max_length-2]
        line = ['<s>'] + line + ['</s>']
        
        line = line + ['<pad>'] * (self.max_length - len(line))
        item = torch.tensor(self.tokenizer.convert_tokens_to_ids(line))
        return item
