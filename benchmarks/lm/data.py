# Copyright (c) 2023  Khaleelulla Khan Nazeer, Anand Subramoney, Mark Schöne, David Kappel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import torch

from collections import Counter
from typing import Tuple


def batchify(data: torch.Tensor, bsz: int, device: torch.device) -> torch.Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size
        device: torch device to load data

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


def get_batch(source: torch.Tensor, i: int, seq_len: int, batch_first: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int starting point in the source tensor
        seq_len: backpropagation through time steps
        batch_first: if True, function return shape (BS, seq_len)

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len]

    # map to shape (BS, SeqLen, ntokens)
    if batch_first:
        data = torch.transpose(data, 0, 1)
        target = torch.transpose(target, 0, 1)
    return data, target.flatten()


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path), f'File does not exist: {path}'
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


def get_data(root: str, dset: str, batch_size: int, device: torch.device):
    """
    Returns Wiki-Text-2 train, val and test split as well as number of tokens

    Args:
        root: directory to store and load the dataset
        dset: choice of WT2 or PTB (Wikitext-2 or PennTreeBank)
        batch_size: batch size
        device: a torch device, e.g. 'cpu' or 'cuda'
    """
    if dset == 'WT2':
        corpus = Corpus(os.path.join(root, 'wikitext-2'))
        print("DATASET: WikiText-2")
    elif dset == 'PTB':
        corpus = Corpus(os.path.join(root, 'PennTreebank'))
        print("DATASET: PennTreebank")
    elif dset == 'WT103':
        corpus = Corpus(os.path.join(root, 'wikitext-103'))
        print("DATASET: WikiText-103")
    else:
        raise NotImplementedError(f'Dataset {dset} not implemented. Choose either WT2 or PTB!')

    train_data = batchify(corpus.train, batch_size, device)
    val_data = batchify(corpus.valid, batch_size, device)
    test_data = batchify(corpus.test, batch_size, device)

    return train_data, val_data, test_data, len(corpus.dictionary)
