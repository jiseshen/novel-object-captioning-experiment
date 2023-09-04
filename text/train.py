from model import *
from data import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '/root/saved_model/'


def train(model, epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for i in range(epoch):
        step = 0
        model.train()
        for caption, length in tqdm(train_loader):
            pack = torch.nn.utils.rnn.pack_padded_sequence(caption, length, batch_first=True).to(device)
            step += 1
            optimizer.zero_grad()
            output = model(pack)
            loss = criterion(output, caption[:, 1:])
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print("After step {} of epoch {}, the training loss is {}".format(step, i, round(loss.item(), 3)))
        model.eval()
        loss = 0
        for caption, length in valid_loader:
            pack = torch.nn.utils.rnn.pack_padded_sequence(caption, length, batch_first=True).to(device)
            output = model(pack)
            loss += criterion(output, caption[:, 1:]).item()
        print("After epoch {}, the validation loss is {}".format(i, round(loss/len(valid_loader), 3)))
        torch.save(model, model_path + 'text{}'.format(i))


text_model = Text(300, 512, len(vocab_map)).to(device)
text_model.word_embed.weight.data.copy_(torch.tensor(glove_emb))
train(text_model, epoch=10)
torch.save(text_model, model_path + 'text')
