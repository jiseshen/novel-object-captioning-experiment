import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class Paired(nn.Module):
    def __init__(self, encoder, decoder, feature, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.word_embed = decoder.word_embed
        self.rnn = decoder.gru
        self.text_trans = decoder.fc
        self.image_trans = nn.Linear(feature, vocab_size)
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        for param in self.word_embed.parameters():
            param.requires_grad_(False)
        for param in self.rnn.parameters():
            param.requires_grad_(False)

    def forward(self, image, caption):
        image_embed = self.image_trans(self.encoder(image))
        text_embed = self.rnn(self.word_embed(caption[:, :-1]))  # ignore the end indicator
        output = self.text_trans(text_embed) + self.image_trans(image_embed)
        return output

    def predict(self, image, max_len=20):
        output = []
        image_embed = self.image_trans(self.image_embed(image))
        rnn_out, hidden = self.rnn(self.word_embed(torch.LongTensor([[0]])))  # start
        outputs = self.text_trans(rnn_out).squeeze(1) + image_embed
        _, max_idx = torch.max(outputs, dim=1)  # greedy
        while not max_idx == 1 and len(output) < max_len:
            output.append(max_idx.cpu()[0].item())
            inputs = self.word_embed(max_idx).unsqueeze(1)
            rnn_out, hidden = self.rnn(inputs, hidden)
            outputs = self.text_trans(rnn_out).squeeze(1) + image_embed
            _, max_idx = torch.max(outputs, dim=1)
        return output
