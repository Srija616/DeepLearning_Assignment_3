
import pandas as pd
from copy import deepcopy
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Vocab:
    def __init__(self, file_path, src_lang, tgt_lang):
        self.data = pd.read_csv(file_path, sep = ',', header = None, names = [src_lang, tgt_lang])
        self.data.dropna()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        src_chars = sorted(set(''.join(self.data[src_lang])))
        tgt_chars = sorted(set(''.join(self.data[tgt_lang])))

        self.src_char_to_idx = {src_chars[i]:i+3 for i in range(len(src_chars))}
        self.src_char_to_idx['<'] = 0
        self.src_char_to_idx['<pad>'] = 1
        self.src_char_to_idx['<unk>'] = 2

        self.src_vocab = {src_chars[i]:i+3 for i in range(len(src_chars))}
        self.src_vocab['<'] = 0
        self.src_vocab['<pad>'] = 1
        self.src_vocab['<unk>'] = 2
        

        
        self.src_idx_to_char = {idx: char for char, idx in self.src_char_to_idx.items()}
 
        self.tgt_char_to_idx =  {tgt_chars[i]:i+3 for i in range(len(tgt_chars))}
        self.tgt_char_to_idx['<'] = 0
        self.tgt_char_to_idx['<pad>'] = 1
        self.tgt_char_to_idx['<unk>'] = 2

        self.tgt_vocab =  {tgt_chars[i]:i+3 for i in range(len(tgt_chars))}
        self.tgt_vocab['<'] = 0
        self.tgt_vocab['<pad>'] = 1
        self.tgt_vocab['<unk>'] = 2


        self.tgt_idx_to_char = {idx: char for char, idx in self.tgt_char_to_idx.items()}
        # self.src_vocab = (self.src_char_to_idx)
        # self.tgt_vocab = (self.tgt_char_to_idx)
        print (len(self.src_vocab), len(self.tgt_vocab))

    def get(self):
        return self.src_vocab, self.tgt_vocab, self.src_char_to_idx, self.src_idx_to_char, self.tgt_char_to_idx, self.tgt_idx_to_char 


class TransliterationDataLoader(Dataset):
    def __init__(self, filename, src_lang, tgt_lang, src_vocab, tgt_vocab, src_char_to_idx, tgt_char_to_idx):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_char_to_idx = src_char_to_idx
        self.tgt_char_to_idx = tgt_char_to_idx
        self.start_of_word = 0
        self.data = pd.read_csv(filename, sep = ',', header = None, names = [self.src_lang, self.tgt_lang])
        
        self.src_dim = max([len(word) for word in self.data[self.src_lang].values]) + 1
        self.tgt_dim = max([len(word) for word in self.data[self.tgt_lang].values]) + 1

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset based on the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple containing the English word and Hindi word.
        """

        # print (self.data.head(5))
        # print (self.data.columns)
        # print (idx)
        src_word = self.data.iloc[idx][self.src_lang]
        tgt_word = self.data.iloc[idx][self.tgt_lang]

        # print (src_word, tgt_word)
        # src_word = self.data.loc[self.src_lang, idx]
        # tgt_word = self.data.loc[self.tgt_lang, idx]

        src_indices = [self.src_vocab.get(char, self.src_vocab['<unk>']) for char in src_word]
        tgt_indices = [self.tgt_vocab.get(char, self.tgt_vocab['<unk>']) for char in tgt_word]

        src_indices.insert(0, self.start_of_word)
        tgt_indices.insert(0, self.start_of_word)
        
        src_len = len(src_indices)
        tgt_len = len(tgt_indices)
        
        src_pad = [self.src_vocab['<pad>']] * (self.src_dim - src_len)
        tgt_pad = [self.tgt_vocab['<pad>']] * (self.tgt_dim - tgt_len)

        src_indices.extend(src_pad)
        tgt_indices.extend(tgt_pad)

        src_tensor = torch.LongTensor(src_indices)
        tgt_tensor = torch.LongTensor(tgt_indices)

        return src_tensor, tgt_tensor, src_len, tgt_len

