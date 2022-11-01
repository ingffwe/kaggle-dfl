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

import torch, gc

gc.collect()
torch.cuda.empty_cache()

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

train_path = glob.glob('./work2/split_images/train/*/*')
# train_path1 = glob.glob('./work2/split_images/train/*/4ffd5986_0*')
# train_path2= glob.glob('./work2/split_images/train/*/9a97dae4_1*')
# train_path3 = glob.glob('./work2/split_images/train/*/407c5a9e_1*')
# train_path4 = glob.glob('./work2/split_images/train/*/ecf251d4_0*')
# train_path1.extend(train_path2)
# train_path1.extend(train_path3)
# train_path1.extend(train_path4)
train_label = [x.split('\\')[-2] for x in train_path]

val_path = glob.glob('./work2/cuted_images/val/*/*')
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

df = pd.concat([train_df,val_df])


'''

tf_efficientnet_b5_ap  50min/epoch


'''
# "model = timm.create_model("tf_efficientnet_b5_ap",
#                           num_classes=4,
#                           in_chans=3,
#                           pretrained=True)"
# model.load_state_dict(torch.load(r'W:\PycharmProjects\kaggle-DFL\dflfiles\tf_efficientnet_b5_ap-456-fix.pt'))


model = torch.jit.load(r'W:\PycharmProjects\kaggle-DFL\dflfiles\finetuned_model.pt','cuda')
# model.load_state_dict(torch.load(r'W:\PycharmProjects\kaggle-DFL\code\ckpt\finetune-49-1.pth'))
# model = torch.jit.load(r'W:\PycharmProjects\kaggle-DFL\code\ckpt_cuted\049_512_.pth')



import albumentations as A
from albumentations.pytorch import ToTensorV2

def train(train_loader, model, criterion, optimizer):
    model.train()
    model = model.to('cuda')
    train_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')

    for i, (input, target) in pbar:
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print("Train loss :"+str(train_loss))
    return train_loss/len(train_loader)


def validate(val_loader, model, criterion):
    # model = model.to('cuda')
    model.eval()
    val_acc = 0.0

    with torch.no_grad():
        # end = time.time()
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Val ')

        for i, (input, target) in pbar:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

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
    train_loader = torch.utils.data.DataLoader(
        DFLDataset(df['path'].values, df['label_int'].values,
                      A.Compose([
                          A.Resize(height=512, width=512),
                          # A.RandomCrop(512,512),
                          A.HorizontalFlip(p=0.5),
                          # A.RandomContrast(p=0.5),
                          # A.RandomBrightness(p=0.5),
                          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                          A.RandomBrightness(limit=2, p=0.2),
                          # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.2),

                          A.OneOf([
                              A.MotionBlur(p=0.2),
                              A.MedianBlur(blur_limit=3, p=0.1),
                              A.Blur(blur_limit=3, p=0.1),
                          ], p=0.2),
                          ToTensorV2(),

                          ])

                      ), batch_size=2, shuffle=True, num_workers=8, pin_memory=False
    )

    val_loader = torch.utils.data.DataLoader(
        DFLDataset(val_df['path'].values, val_df['label_int'].values,
                      A.Compose([
                          A.Resize(height=384, width=512),
                          # A.CenterCrop(512,512),
                          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                          ToTensorV2(),
                          ])
                      ), batch_size=1, shuffle=False, num_workers=8, pin_memory=False
    )


    # model = model.to('cuda')
    criterion = nn.CrossEntropyLoss().cuda()  # 自带softmax
    # optimizer = torch.optim.SGD(model.parameters(), 0.005)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=5, mode="triangular2")

    optimizer = torch.optim.Adam(model.parameters(),lr=3e-5)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00003, max_lr=0.0003, step_size_up=5, mode="triangular2")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)

    # best_acc = validate(val_loader, model, criterion)
    best_acc = 0.5
    print(best_acc)
    best_loss = 2
    for _ in range(50):
        print('epoch:' + str(_ + 1))

        train_loss = train(train_loader, model, criterion, optimizer)

        # val_acc = validate(val_loader, model, criterion)
            # print(val_acc)
        val_acc = 0
        # if best_acc < val_acc:
        #     # torch.save(model.state_dict(), './ckpt_cuted/finetune-528.pth')
        #     traced = torch.jit.trace(model.cpu(), torch.rand(1, 3, 384, 512))
        #     traced.save(f'./ckpt_cuted/banchang_512.pth')
        #     best_acc = val_acc

        if train_loss < best_loss:
            # torch.save(model.state_dict(), './ckpt_cuted/finetune-528-best-loss.pth')
            traced = torch.jit.trace(model.cpu(), torch.rand(1, 3, 384, 512))
            traced.save(f'./ckpt/finally.pth')
            best_loss = train_loss

        scheduler.step()
        print(train_loss, val_acc)