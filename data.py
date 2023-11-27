import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from biotite.sequence.io.fasta import FastaFile 
from tokenizers import Tokenizer


def cycle(loader):
    while True:
        for data in loader:
            yield data


class ProteinDataset(Dataset):

    def __init__(self, proteins, chars, max_word_length):
        self.proteins = proteins
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.proteins)

    def contains(self, word):
        return word in self.proteins

    def get_vocab_size(self):
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token followed by proteins

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.proteins[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y


class BpeProteinDataset(Dataset):
    """Dataset of protein sequences that is encoded using a byte-pair encoding"""

    def __init__(self, proteins, tokenizer, max_word_length):
        self.proteins = proteins
        self.tokenizer = tokenizer 
        self.max_word_length = max_word_length

    def __len__(self):
        return len(self.proteins) 

    def contains(self, word):
        return word in self.proteins 

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token followed by proteins

    def encode(self, word):
        return torch.tensor(self.tokenizer.encode(word).ids, dtype=torch.long)

    def decode(self, ix):
        return self.tokenizer.decode(ix).tokens

    def __getitem__(self, idx):
        word = self.proteins[idx]
        ix = self.encode(word) 
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1   # index -1 will mask the loss at inactive locations 
        return x, y


def create_datasets(input_file, split_ratio=0.1, min_sequence_length=32, max_sequence_length=512, tokenizer=None):
    """Create the `train_dataset` and `test_dataset`"""

    # preprocessing of the input text file
    proteins = []
    fasta_file = FastaFile.read(input_file) 
    for header, sequence in fasta_file.items():
        n = len(sequence) 
        if n >= min_sequence_length and n < max_sequence_length:
            proteins.append(sequence)
    max_word_length = max(len(w) for w in proteins)

    # partition the input data into a training and the test set
    test_set_size = min(1000, int(len(proteins) * split_ratio)) 
    rp = torch.randperm(len(proteins)).tolist()
    train_proteins = [proteins[i] for i in rp[:-test_set_size]]
    test_proteins = [proteins[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_proteins)} training examples and {len(test_proteins)} test examples")

    if not tokenizer:
        chars = sorted(list(set(''.join(proteins)))) # all the possible characters
        tokens = sum(len(w) for w in proteins)
        print("using characters as tokens")
        print(f"number of examples in the dataset: {len(proteins)}")
        print(f"max protein length: {max_word_length}")
        print(f"number of unique characters in the vocabulary: {len(chars)}")
        print("vocabulary:")
        print(''.join(chars))
        print(f"total tokens: {tokens}")

        # wrap in dataset objects
        train_dataset = ProteinDataset(train_proteins, chars, max_word_length)
        test_dataset = ProteinDataset(test_proteins, chars, max_word_length)
    else:
        print(f"using tokenizer from: {tokenizer}") 
        tokenizer = Tokenizer.from_file(tokenizer)
        train_dataset = BpeProteinDataset(train_proteins, tokenizer, max_word_length) 
        test_dataset = BpeProteinDataset(test_proteins, tokenizer, max_word_length) 

    return train_dataset, test_dataset