import nltk
import pickle
import json
import os
import random
from tqdm import tqdm
import numpy as np
import torch
from heapq import nlargest
from collections import Counter
from torch.utils.data import DataLoader, Dataset, sampler
import torchtext.vocab as vocab
from pycocotools.coco import COCO


train_annotation_path = '/root/coco/annotations/captions_train2014.json'
train_coco = COCO(train_annotation_path)


"""
Build Vocab
"""

count = Counter()
ann_sum = {}

last_image = None
for ann_index in train_coco.anns:
    text = str(train_coco.anns[ann_index]['caption']).lower()
    token = nltk.tokenize.word_tokenize(text)
    count.update(token)
    image_id = train_coco.anns[ann_index]['image_id']
    if image_id not in ann_sum:
        ann_sum[image_id] = 0
    for i in token:
        ann_sum[image_id] += 1
vocab_map = {"<BOS>": 0, "<EOS>": 1, "<UNK>": 2}
vocab_list = ["<BOS>", "<EOS>", "<UNK>"]
for word, num in count.items():
    if num >= 8:  # Omit words with frequency less than 8
        vocab_map[word] = len(vocab_map)
        vocab_list.append(word)

glove = vocab.GloVe(name='6B', dim=300)  # Get GloVe embedding
glove_emb = np.zeros((len(vocab_list), 300))
for word in vocab_map:
    if word in glove.stoi:
        glove_emb[vocab_map[word]] = glove.vectors[glove.stoi[word]]
    else:
        glove_emb[vocab_map[word]] = np.random.rand(300) * 2 - 1

with open("/root/vocab.pkl", "wb") as f:
    pickle.dump([vocab_map, vocab_list, torch.tensor(glove_emb, requires_grad=True)], f)


"""
Build Lexicon
"""

tf = {word: 0 for word in vocab_map if word not in ('<BOS>', '<EOS>', '<UNK>')}
df = {word: 0 for word in tf}
df_temp = {word: 0 for word in tf}
tf_temp = {word: 0 for word in tf}
last_image = None
for ann_index in train_coco.anns:
    text = str(train_coco.anns[ann_index]['caption']).lower()
    token = nltk.tokenize.word_tokenize(text)
    image_id = train_coco.anns[ann_index]['image_id']
    if not image_id == last_image:
        df = {word: df[word] + df_temp[word] for word in tf}
        tf = {word: max(tf_temp[word], tf[word]) for word in tf}
        df_temp = {word: 0 for word in tf}
        tf_temp = {word: 0 for word in tf}
        last_image = image_id
    for word in token:
        if word in tf:
            tf_temp[word] += 1/ann_sum[image_id]
            df_temp[word] = 1

rank = {word: tf[word] * np.log(len(train_coco.anns) / 5 / df[word]) for word in tf}
lexicon_list = nlargest(500, rank, key=lambda x: rank[x])
lexicon_map = {lexicon: lexicon_list.index(lexicon) for lexicon in lexicon_list}

with open("lexicon_range.pkl", "wb") as f:
    pickle.dump([lexicon_list, lexicon_map], f)


