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
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO

"""
Data Read-in
"""

novel_objects = ['bottle', 'bus', 'couch', 'microwave', 'pizza', 'racket', 'suitcase', 'zebra']

novel_objects_with_plural = ['zebra', 'zebras', 'pizza', 'pizzas', 'suitcase', 'suitcases', 'luggage', 'luggages',
                             'bottle', 'bottles', 'bus', 'busses', 'couch', 'couches', 'microwave', 'microwaves',
                             'racket', 'rackets']

paired_train_path = '/root/coco/annotations/captions_no_caption_rm_eightCluster_train2014.json'
train_coco = COCO(paired_train_path)
paired_valid_path = '/root/coco/annotations/captions_val_val2014.json'
valid_coco = COCO(paired_valid_path)

test_paths = ['/root/coco/annotations/captions_split_set_{}_val_test_novel2014.json'.format(novel_object) for
              novel_object in novel_objects]
test_cocos = {}
for novel_object, test_path in zip(novel_objects, test_paths):
    test_coco = COCO(test_path)
    test_cocos[novel_object] = test_coco

image_path = '/root/coco/images/'

with open("/root/vocab.pkl", "rb") as f:
    [vocab_map, vocab_list, glove_emb] = pickle.load(f)

with open('/root/lexicon_range.pkl', 'rb') as f:
    [lexicon_list, lexicon_map] = pickle.load(f)

"""
Customize Dataset
"""


class CocoDataset(Dataset):
    def __init__(self, coco, test=False):
        self.coco = coco
        self.test = test
        self.index_list = list(self.coco.anns.keys())
        self.transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor()])
        for ann in self.index_list:
            self.coco.anns[ann]['caption'] = nltk.tokenize.word_tokenize(self.coco.anns[ann]['caption'].lower())
        self.caption_lengths = [len(self.coco.anns[ann]['caption']) for ann in
                                self.index_list]

    def __getitem__(self, index):
        ann_info = self.coco.anns[self.index_list[index]]

        caption = ann_info['caption']
        caption = [0] + [caption] + [1]

        img_index = ann_info['image_id']
        img_path = image_path + self.coco.loadImgs(img_index)[0]['file_name']
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        if self.test:
            return image, caption, img_index
        return image, caption

    def __len__(self):
        return len(self.index_list)

    def random_length(self):  # Randomly choose a caption length to unify the length within batch
        length = random.sample(self.caption_lengths, 1)
        selected = np.where([self.caption_lengths[i] == length for i in range(len(self.caption_lengths))])[0]
        return selected


seed = 42
torch.random.manual_seed(seed)

batch_size = 64
train_set = CocoDataset(train_coco)
train_loader = DataLoader(train_set,
                          batch_sampler=sampler.BatchSampler(
                              sampler.SubsetRandomSampler(train_set.random_length()), batch_size, drop_last=False))

valid_set = CocoDataset(valid_coco)
valid_loader = DataLoader(valid_set,
                          batch_sampler=sampler.BatchSampler(
                              sampler.SubsetRandomSampler(valid_set.random_length()), batch_size, drop_last=False))

test_sets = {o: CocoDataset(test_cocos[o], test=True) for o in test_cocos}
test_loaders = {o: DataLoader(test_sets[o], batch_size=1, shuffle=False) for o in test_sets}
