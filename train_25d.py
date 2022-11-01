from datetime import time

from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import timm

import os
import sys
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

import cv2
from PIL import Image

import torch
torch.manual_seed(0)  # 减少随机性
torch.backends.cudnn.deterministic = False  # 是否有确定性
torch.backends.cudnn.benchmark = True  # 自动寻找最适合当前配置的高效算法，提高运行效率


class CFG:
    # step1: hyper-parameter
    seed = 42  # birthday
    num_worker = 16  # debug => 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_fold = "ckpt-frank"

    # step2: data
    n_25d_shift = 2
    n_fold = 4
    img_size = (456, 456)
    train_bs = 128
    valid_bs = train_bs * 2


    # step4: optimizer
    epoch = 12
    lr = 1e-3
    wd = 1e-5
    lr_drop = 8

class DFLDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):


        # img = cv2.imread(self.img_path[index])
        img_path  = self.img_path[index]
        img = self.load_3d_slice(img_path) # [h, w, c]

        img = img.astype(np.float32)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if self.transform is not None:
            img = self.transform(image=img)['image']
        return img, torch.from_numpy(np.array(self.img_label[index]))

    def __len__(self):
        return len(self.img_path)

    def load_3d_slice(self, middle_img_path):
        #### 步骤1: 获取中间图片的基本信息
        #### eg: middle_img_path: './work_25d/split_images/train\play\1606b0e6_1_012923_mid.jpg'
        middle_slice = os.path.basename(middle_img_path).split('_mid')[0].split('\\')[-1].split('.jpg')[0] # eg: 1606b0e6_1_012923
        middle_slice_num = middle_slice.split('_')[-1]

        new_25d_imgs = []

        ##### 步骤2：按照左右n_25d_shift数量进行填充，如果没有相应图片填充为Nan.
        ##### 注：经过EDA发现同一天的所有患者图片的shape是一致的
        for i in range(-3, 4):  # eg: i = {-2, -1, 0, 1, 2}
            if i != 0:
                shift_slice_num = int(middle_slice_num) + i
                shift_slice_str = str(shift_slice_num).zfill(6)
                shift_img_path = middle_img_path.replace(middle_slice_num + '_mid', shift_slice_str)
            else:
                shift_img_path = middle_img_path
            if os.path.exists(shift_img_path):
                shift_img = cv2.imread(shift_img_path, cv2.IMREAD_UNCHANGED)  # [w, h]
                shift_img = cv2.cvtColor(shift_img, cv2.COLOR_RGB2GRAY)
                # shift_img = cv2.resize(shift_img,CFG.img_size)

                new_25d_imgs.append(shift_img)
            else:
                new_25d_imgs.append(None)
                # print(shift_img_path)

        ##### 步骤3：从中心开始往外循环，依次填补None的值
        ##### eg: n_25d_shift = 2, 那么形成5个channel, idx为[0, 1, 2, 3, 4], 所以依次处理的idx为[1, 3, 0, 4]
        shift_left_idxs = []
        shift_right_idxs = []
        for related_idx in range(3):
            shift_left_idxs.append(2 - related_idx)
            shift_right_idxs.append(3 + related_idx + 1)

        for left_idx, right_idx in zip(shift_left_idxs, shift_right_idxs):
            if new_25d_imgs[left_idx] is None:
                new_25d_imgs[left_idx] = new_25d_imgs[3]
            if new_25d_imgs[right_idx] is None:
                new_25d_imgs[right_idx] = new_25d_imgs[3]

        new_25d_imgs = np.stack(new_25d_imgs, axis=2).astype('float32')  # [w, h, c]
        mx_pixel = new_25d_imgs.max()
        if mx_pixel != 0:
            new_25d_imgs /= mx_pixel
        return new_25d_imgs

train_path = glob.glob('./work_25d/split_images/train/*/*mid*')
train_label = [x.split('\\')[-2] for x in train_path]

val_path = glob.glob('./work_25d/split_images/val/*/*mid*')
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
# train_df = train_df.sample(frac=1.0)


model = timm.create_model("tf_efficientnet_b5_ap",
                          num_classes=4,
                          in_chans=7,
                          pretrained=False)

# model.load_state_dict(torch.load('./ckpt/ENB5_25d_456.pth'))

import albumentations as A
from albumentations.pytorch import ToTensorV2

def train(train_loader, model, criterion, optimizer):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
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
        # print(loss.item())
        train_loss += loss.item()
        train_acc += (output.argmax(1) == target).sum().item()

    print('----------train loss--------------')
    print(train_loss/len(train_loader))
    print('----------train acc--------------')
    print(train_acc/len(train_loader))

    return train_loss/len(train_loader)


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


# 随机拆分
train_loader = torch.utils.data.DataLoader(
    DFLDataset(train_df['path'].values, train_df['label_int'].values,
                  A.Compose([
                      A.Resize(456, 456),
                      A.HorizontalFlip(p=0.5),
                      # A.RandomContrast(p=0.5),
                      # A.RandomBrightness(p=0.5),
                      # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                      ToTensorV2(),
                      ])

                  ), batch_size=2, shuffle=True, num_workers=10, pin_memory=False
)

val_loader = torch.utils.data.DataLoader(
    DFLDataset(val_df['path'].values, val_df['label_int'].values,
                  A.Compose([
                      A.Resize(456, 456),
                      # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                      ToTensorV2(),
                      ])
                  ), batch_size=2, shuffle=False, num_workers=10, pin_memory=False
)

if __name__ == '__main__':
    model = model.to('cuda')
    criterion = nn.CrossEntropyLoss().cuda()  # 自带softmax
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=5, mode="triangular2")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)

    best_loss = 2
    for _ in range(15):
        print('epoch:'+str(_+1))

        train_loss = train(train_loader, model, criterion, optimizer)
        val_acc = validate(val_loader, model, criterion)

        if train_loss < best_loss:
            torch.save(model.state_dict(), 'ckpt/resnet18_25d.pth')
            best_loss = train_loss

        scheduler.step()
        print(_+1)
        print(train_loss, val_acc)