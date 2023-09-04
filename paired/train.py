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
        for image, caption in tqdm(train_loader):
            step += 1
            image.to(device)
            image = train_pre(image)
            caption = caption.to(device)
            optimizer.zero_grad()
            output = model(image, caption)
            loss = criterion(output, caption[:, 1:])
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print("After step {} of epoch {}, the training loss is {}".format(step, i, round(loss.item(), 3)))
        model.eval()
        loss = 0
        for image, caption in valid_loader:
            image = image.to(device)
            caption = caption.to(device)
            output = model(image, caption)
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


def transit(model):
    model.cpu()
    for o in novel_objects_with_plural:
        o_id = vocab_map[o]
        o_emb = glove_emb[o_id]
        min_dist = float("inf")
        closest_id = None
        for i, emb in enumerate(glove_emb):
            if vocab_list[i] not in lexicon_map or i == o_id:
                continue
            dist = np.linalg.linalg.norm(o_emb-emb)
            if dist < min_dist:
                min_dist = dist
                closest_id = i
        closest_lexicon_index = lexicon_map[vocab_list[closest_id]]
        with torch.no_grad():
            image_weight, image_bias = model.image_trans.parameters()
            text_weight, text_bias = model.text_trans.parameters()
            for param in [image_weight, image_bias, text_weight, text_bias]:
                param[o_id] = param[closest_id]
            image_weight[o_id, lexicon_map[o]] = image_weight[closest_id, closest_lexicon_index]
            image_weight[o_id, closest_lexicon_index] = 0
            image_weight[closest_id, lexicon_map[o]] = 0
    model.to(device)


lexicon = torch.load(model_path + 'lexicon')
text = torch.load(model_path, 'text')
paired_model = Paired(lexicon, text, len(lexicon_list), len(vocab_list)).to(device)
train(paired_model, epoch=10)
torch.save(paired_model, model_path + 'paired')
evaluate(paired_model)
