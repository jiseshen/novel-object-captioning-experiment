import torch
from torch import nn
from torchvision.models import resnet50


class NOC(nn.Module):
    def __init__(self, embed_dim, hidden_size, vocab_size, glove_emb, num_layers=1):
        super().__init__()
        self.fc = torch.tensor(glove_emb, requires_grad=True)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.image_embed = nn.Sequential(*list(resnet50().children())[:-1])
        self.im_fc = nn.Linear(2048, vocab_size)
        self.im_fc.weight.data.copy_(glove_emb)

    def lexicon(self, image):
        embed = self.image_embed(image)
        return self.im_fc(embed)

    def text(self, captions):
        embed = self.fc.T * torch.nn.functional.one_hot(captions)
        outputs, _ = self.gru(embed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=captions.shape[0])
        outputs = self.fc * outputs
        return outputs

    def paired(self, image, captions):
        return self.lexicon(image) + self.text(captions)

    def predict(self, image, max_len=20):
        output = []
        image_embed = self.lexicon(image)
        rnn_out, hidden = self.gru(self.word_embed(torch.LongTensor([[0]])))
        outputs = self.im_fc(rnn_out).squeeze(1) + image_embed
        _, max_idx = torch.max(outputs, dim=1)
        while not max_idx == 1 and len(output) < max_len:
            output.append(max_idx.cpu()[0].item())
            inputs = self.word_embed(max_idx).unsqueeze(1)
            rnn_out, hidden = self.rnn(inputs, hidden)
            outputs = self.text_trans(rnn_out).squeeze(1) + image_embed
            _, max_idx = torch.max(outputs, dim=1)
        return output