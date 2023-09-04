from model import *
from data import *
from pycocoevalcap.eval import COCOEvalCap


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '/root/saved_model/'


train_pre = nn.Sequential(
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
)


def train(model, epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for i in range(epoch):
        step = 0
        model.train()
        for image, caption, length in tqdm(paired_train_loader):
            step += 1
            image.to(device)
            image = train_pre(image)
            caption = caption.to(device)
            optimizer.zero_grad()
            pack = torch.nn.utils.rnn.pack_padded_sequence(caption, length, batch_first=True).to(device)
            output = model.paired(image, pack)
            paired_loss = criterion(output, caption[:, 1:])  # since padding is -100, the out-of-bound loss will not be calculated
            image, label = next(iter(lexicon_train_loader))
            image = train_pre(image.to(device))
            output = model.lexicon(image)
            lexicon_loss = criterion(output, label.to(device))
            caption, length = next(iter(text_train_loader))
            pack = torch.nn.utils.rnn.pack_padded_sequence(caption, length, batch_first=True).to(device)
            output = model.text(pack)
            text_loss = criterion(output, caption[:, 1:])
            loss = paired_loss + lexicon_loss + text_loss
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print("After step {} of epoch {}, the training loss is {}".format(step, i, round(loss.item(), 3)))
        model.eval()
        loss = 0
        for image, caption in valid_loader:
            image = image.to(device)
            caption = caption.to(device)
            output = model.paired(image, caption)
            loss += criterion(output, caption[:, 1:]).item()
        print("After epoch {}, the validation loss is {}".format(i, round(loss/len(valid_loader), 3)))
        torch.save(model, model_path + 'paired{}'.format(i))


def evaluate(model):
    model.eval()
    for o in test_loaders:
        tp = 0
        fp = 0
        fn = 0
        results = []
        id_occ = {}
        for image, caption, img_index in test_loaders[o]:
            image = image.to(device)
            predicted = model.predict(image)
            predicted = ' '.join([vocab_list[index] for index in predicted[:-1]])
            if o in caption and o in predicted:
                tp += 1
            if o not in caption and o in predicted:
                fp += 1
            if o in caption and o not in predicted:
                fn += 1
            results.append({"image_id": int(img_index.item()), "caption": predicted})
            id_occ[str(img_index)] = 1
        with open('results', 'w') as temp_f:
            json.dump(results, temp_f)
        coco = test_cocos[o]
        coco_result = coco.loadRes('results')
        cocoEval = COCOEvalCap(coco, coco_result)
        cocoEval.params['image_id'] = coco_result.getImgIds()
        cocoEval.evaluate()
        print(o + ":")
        for metric, score in cocoEval.eval.items():
            print(metric + ": {}".format(round(score, 3)))
        print('F1: {}'.format(round(2*tp/(2*tp+fn+fp))))


lexicon = torch.load(model_path + 'lexicon')
text = torch.load(model_path, 'text')
paired_model = NOC(lexicon, text, len(vocab_list), glove_emb).to(device)
train(paired_model, epoch=10)
torch.save(paired_model, model_path + 'paired')
evaluate(paired_model)
