import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
from IPython.display import Video
import cv2
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import timm
from timm.data import ImageDataset, create_loader, resolve_data_config
import os
import sys
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from timm.utils import AverageMeter, setup_default_logging
# %pylab inline

import cv2
from PIL import Image

import torch

torch.manual_seed(0)  # 减少随机性
torch.backends.cudnn.deterministic = False  # 是否有确定性
torch.backends.cudnn.benchmark = True  # 自动寻找最适合当前配置的高效算法，提高运行效率

class DFLDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = cv2.imread(self.img_path[index])
        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)['image']
        return img, torch.from_numpy(np.array(self.img_label[index]))

    def __len__(self):
        return len(self.img_path)

train_path = glob.glob('./work/split_images/train/*/*')
train_label = [x.split('\\')[-2] for x in train_path]

val_path = glob.glob('./work/split_images/val/*/*')
val_label = [x.split('\\')[-2] for x in val_path]

train_df = pd.DataFrame({
    'path': train_path,
    'label': train_label
})
train_df['label_int'], lbl = pd.factorize(train_df['label'])
lbl = list(lbl)
train_df = train_df.sample(frac=1.0)

val_df = pd.DataFrame({
    'path': val_path,
    'label': val_label
})
val_df['label_int'] = val_df['label'].apply(lambda x: lbl.index(x))
train_df = train_df.sample(frac=1.0)

# model = models.resnet18(True)
# model.fc = nn.Sequential(
#     nn.Dropout(p=0.5, inplace=True),
#     nn.Linear(in_features=512, out_features=4, bias=True)
# )
model = timm.create_model("tf_efficientnet_b5_ap",
                          num_classes=4,
                          in_chans=3,
                          pretrained=True)

model.load_state_dict(torch.load(r'W:\PycharmProjects\kaggle-DFL\dflfiles\tf_efficientnet_b5_ap-456.pt'))

import albumentations as A
from albumentations.pytorch import ToTensorV2


def validate(val_loader, model, criterion):
    model.eval()

    val_acc = 0.0

    with torch.no_grad():
        # end = time.time()
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Val ')

        for i, (input, target) in pbar:
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            val_acc += (output.argmax(1) == target).sum().item()

    return val_acc / len(val_loader.dataset)


def predict(test_loader, model, criterion):
    model.eval()
    val_acc = 0.0

    test_pred = []
    with torch.no_grad():
        # end = time.time()
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            test_pred.append(output.data.cpu().numpy())

    return np.vstack(test_pred)

if __name__ == '__main__':
# 随机拆分

    val_loader = torch.utils.data.DataLoader(
        DFLDataset(val_df['path'].values, val_df['label_int'].values,
                      A.Compose([
                          A.Resize(456, 456),
                          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                          ToTensorV2(),
                          ])
                      ), batch_size=2, shuffle=False, num_workers=10, pin_memory=False
    )


    model = model.to('cuda')
    criterion = nn.CrossEntropyLoss().cuda()  # 自带softmax
    optimizer = torch.optim.SGD(model.parameters(), 0.005)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=5, mode="triangular2")

    best_loss = 2
    for _ in range(1):
        print('epoch:' + str(_ + 1))

        val_acc = validate(val_loader, model, criterion)

        scheduler.step()
        print(val_acc)