from torch import nn
from torchvision import models


class Lexicon(nn.Module):
    def __init__(self, model, encode_dim, lexicon_num):
        super().__init__()
        self.encoder = nn.Sequential(*list(model.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(encode_dim, lexicon_num)

    def forward(self, image):
        emb = self.flatten(self.encoder(image))
        logit = self.fc(emb)
        return logit

