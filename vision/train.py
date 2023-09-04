from model import *
from data import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '/root/saved_model/'


train_pre = nn.Sequential(
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
)


def train(model, epoch):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-1, epochs=epoch, steps_per_epoch=len(train_loader))
    for i in range(epoch):
        step = 0
        model.train()
        for image, lexicon in tqdm(train_loader):
            image = image.to(device)
            image = train_pre(image)
            lexicon = lexicon.to(device)
            step += 1
            optimizer.zero_grad()
            logit = model(image)
            loss = criterion(logit, lexicon)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 100 == 0:
                print("After step {} of epoch {}, the training loss is {}".format(step, i, round(loss.item(), 3)))
        model.eval()
        loss = 0
        for image, lexicon in valid_loader:
            image = image.to(device)
            lexicon = lexicon.to(device)
            logit = model(image)
            loss += criterion(logit, lexicon).item()
        print("After epoch {}, the validation loss is {}".format(i, round(loss, 3)))
        torch.save(model, model_path + 'lexicon{}'.format(i))


lexicon_model = Lexicon(models.resnet50(), 2048, len(lexicon_list)).to(device)
train(lexicon_model, epoch=10)
torch.save(lexicon_model, model_path + 'lexicon')
