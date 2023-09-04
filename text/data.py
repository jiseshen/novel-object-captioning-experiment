import nltk
import pickle
import json
import os
import random
from tqdm import tqdm
import numpy as np
import torch
from collections import Counter
from torch.utils.data import DataLoader, Dataset, sampler
import torchtext.vocab as vocab
from pycocotools.coco import COCO

"""
Data Read-in
"""

train_annotation_path = '/root/coco/annotations/captions_train2014.json'
train_coco = COCO(train_annotation_path)
valid_annotation_path = '/root/coco/annotations/captions_val2014.json'
valid_coco = COCO(valid_annotation_path)

with open("/root/vocab.pkl", "rb") as f:
    [vocab_map, vocab_list, glove_emb] = pickle.load(f)


def get_index(w):
    if w in vocab_map:
        return vocab_map[w]
    else:
        return 2  # <UNK>


"""
Dataset
"""


class TextDataset(Dataset):
    def __init__(self, coco):
        self.coco = coco
        self.index_list = list(self.coco.anns.keys())
        self.caption_lengths = [len(nltk.tokenize.word_tokenize(self.coco.anns[ann]['caption'].lower())) for ann in
                                self.index_list]

    def __getitem__(self, index):
        ann_info = self.coco.anns[self.index_list[index]]

        caption = str(ann_info['caption']).lower()
        caption = [0] + [get_index(w) for w in nltk.tokenize.word_tokenize(caption)] + [1]

        return caption

    def __len__(self):
        return len(self.index_list)


def rnn_collate(data):
    data.sort(key=lambda x: len(x), reverse=True)
    lengths = [len(seq[0]) for seq in data]
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=-100)
    return torch.LongTensor(data), lengths


seed = 42
torch.random.manual_seed(seed)

train_loader = DataLoader(TextDataset(train_coco), batch_size=64, shuffle=True, collate_fn=rnn_collate, num_workers=12)
valid_loader = DataLoader(TextDataset(valid_coco), batch_size=64, shuffle=False, collate_fn=rnn_collate, num_workers=12)
