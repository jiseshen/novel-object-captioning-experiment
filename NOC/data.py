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

image_path = '/root/coco/images/'

train_annotation_path = '/root/coco/annotations/captions_train2014.json'
train_coco = COCO(train_annotation_path)
valid_annotation_path = '/root/coco/annotations/captions_val2014.json'
valid_coco = COCO(valid_annotation_path)


paired_train_path = '/root/coco/annotations/captions_no_caption_rm_eightCluster_train2014.json'
paired_train_coco = COCO(paired_train_path)

test_paths = ['/root/coco/annotations/captions_split_set_{}_val_test_novel2014.json'.format(novel_object) for
              novel_object in novel_objects]
test_cocos = {}
for novel_object, test_path in zip(novel_objects, test_paths):
    test_coco = COCO(test_path)
    test_cocos[novel_object] = test_coco

with open("/root/vocab.pkl", "rb") as f:
    [vocab_map, vocab_list, glove_emb] = pickle.load(f)

def get_index(w):
    if w in vocab_map:
        return vocab_map[w]
    else:
        return 2  # <UNK>

"""
Build Lexicon
"""

if os.path.exists('/root/lexicon_NOC.pkl'):
    with open('/root/lexicon_NOC.pkl', 'rb') as f:
        [train_lexicon, valid_lexicon] = pickle.load(f)

else:
    train_lexicon = {}
    for ann in train_coco.anns:
        text = str(train_coco.anns[ann]['caption']).lower()
        token = nltk.tokenize.word_tokenize(text)
        img_id = train_coco.anns[ann]['image_id']
        if img_id not in train_lexicon:
            train_lexicon[img_id] = np.zeros(len(vocab_map))
        for word in token:
            if word in vocab_map:
                train_lexicon[img_id][vocab_map[word]] = 1

    train_lexicon = [(train_lexicon[image], image) for image in train_lexicon]

    valid_lexicon = {}
    for ann in valid_coco.anns:
        text = str(valid_coco.anns[ann]['caption']).lower()
        token = nltk.tokenize.word_tokenize(text)
        img_id = valid_coco.anns[ann]['image_id']
        if img_id not in valid_lexicon:
            valid_lexicon[img_id] = np.zeros(len(vocab_map))
        for word in token:
            if word in vocab_map:
                valid_lexicon[img_id][vocab_map[word]] = 1

    valid_lexicon = [(valid_lexicon[image], image) for image in valid_lexicon]

    with open('/root/lexicon_NOC.pkl', 'wb') as f:
        pickle.dump([train_lexicon, valid_lexicon], f)

"""
Three Datasets
"""

class NOCLexicon(Dataset):
    def __init__(self, lexicon):
        self.transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor()])
        self.lexicon_list = lexicon

    def __getitem__(self, index):
        img_path = self.lexicon_list[index][1]
        image = Image.open('/root/coco/images/train2014/'+img_path).convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(self.lexicon_list[index][0])

    def __len__(self):
        return len(self.lexicon_list)


class NOCText(Dataset):
    def __init__(self, coco):
        self.coco = coco
        self.index_list = list(self.coco.anns.keys())

    def __getitem__(self, index):
        ann_info = self.coco.anns[self.index_list[index]]

        caption = str(ann_info['caption']).lower()
        caption = [0] + [get_index(w) for w in nltk.tokenize.word_tokenize(caption)] + [1]

        return caption

    def __len__(self):
        return len(self.index_list)


class NOCPaired(Dataset):
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


def rnn_collate1(data):
    data.sort(key=lambda x: len(x), reverse=True)
    lengths = [len(seq) for seq in data]
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=-100)
    return torch.LongTensor(data), lengths


def rnn_collate2(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lengths = [len(seq[0]) for seq in data]
    image = [seq[1] for seq in data]
    data = [seq[0] for seq in data]
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=-100)
    return torch.tensor(image), torch.LongTensor(data), lengths


seed = 42
torch.random.manual_seed(seed)

batch_size = 64
train_set = NOCLexicon(train_lexicon)
lexicon_train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=12)

train_set = NOCText(train_coco)
text_train_loader = DataLoader(train_set, batch_size, shuffle=True, collate_fn=rnn_collate1)

train_set = NOCPaired(paired_train_coco)
paired_train_loader = DataLoader(train_set, batch_size, shuffle=True, collate_fn=rnn_collate2)

valid_set = NOCPaired(valid_coco)
valid_loader = DataLoader(valid_set, batch_size, shuffle=False, collate_fn=rnn_collate2)

test_sets = {o: NOCPaired(test_cocos[o]) for o in test_cocos}
test_loaders = {o: DataLoader(test_sets[o], batch_size=1, shuffle=False) for o in test_sets}
