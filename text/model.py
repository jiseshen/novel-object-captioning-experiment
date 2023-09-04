import torch
from torch import nn


class Text(nn.Module):
    def __init__(self, embed_dim, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, captions):
        embed = self.word_embed(captions)
        outputs, _ = self.gru(embed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=captions.shape[0])
        outputs = self.fc(outputs)
        return outputs
