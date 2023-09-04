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

train_path = '/root/coco/annotations/captions_train2014.json'
train_coco = COCO(train_path)
valid_path = '/root/coco/annotations/captions_val2014.json'
valid_coco = COCO(valid_path)

"""
Build Lexicon
"""

if os.path.exists('/root/lexicon.pkl'):
    with open('/root/lexicon.pkl', 'rb') as f:
        [train_lexicon, valid_lexicon] = pickle.load(f)

else:

    class CountSet:  # Only set those lexicons with greater than 2 count as positive
        def __init__(self):
            self.label = {}
            self.count = {}

        def update(self, lexicon, image):
            if image not in self.label:
                self.label[image] = np.zeros(471)
            if self.label[image][lexicon] == 1:
                return
            if (lexicon, image) not in self.count:
                self.count[(lexicon, image)] = 1
            else:
                self.count[(lexicon, image)] += 1
            if self.count[(lexicon, image)] > 2:
                self.label[image][lexicon] = 1


    with open('/root/lexicon_range.pkl', 'rb') as f:
        lexicon_list, lexicon_map = pickle.load(f)

    train_lexicon = CountSet()

    for ann in train_coco.anns:
        text = str(train_coco.anns[ann]['caption']).lower()
        token = nltk.tokenize.word_tokenize(text)
        for word in token:
            if word in lexicon_map:
                train_lexicon.update(lexicon_map[word], train_coco.loadImgs(train_coco.anns[ann]['image_id'])[0]['file_name'])

    train_lexicon = [(train_lexicon.label[image], image) for image in train_lexicon.label]

    valid_lexicon = CountSet()

    for ann in train_coco.anns:
        text = str(train_coco.anns[ann]['caption']).lower()
        token = nltk.tokenize.word_tokenize(text)
        for word in token:
            if word in lexicon_map:
                valid_lexicon.update(lexicon_map[word], train_coco.loadImgs(train_coco.anns[ann]['image_id'])[0]['file_name'])

    valid_lexicon = [(valid_lexicon.label[image], image) for image in valid_lexicon.label]

    with open('/root/lexicon.pkl', 'wb') as f:
        pickle.dump([train_lexicon, valid_lexicon], f)

"""
Customize Dataset
"""


class LexiconDataset(Dataset):
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


seed = 42
torch.random.manual_seed(seed)

batch_size = 64
train_set = LexiconDataset(train_lexicon)
train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=12)
valid_set = LexiconDataset(valid_lexicon)
valid_loader = DataLoader(valid_set, batch_size, shuffle=False, num_workers=12)
