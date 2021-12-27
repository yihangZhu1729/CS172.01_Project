from numpy import concatenate
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataLoader import *
import math
from CBAM import *

# some settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
learning_rate = 0.01
max_epochs = 1
train_batch = 1
thresh = 0.5


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.out_channels = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.out_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.mp1 = nn.MaxPool2d(4, stride=4)
        self.mp2 = nn.MaxPool2d(2, stride=2)

        self.attention1 = CBAM(self.out_channels)
        self.attention2 = CBAM(self.out_channels)
        self.attention3 = CBAM(self.out_channels)

        self.classifier = nn.Sequential(
            nn.Linear(self.out_channels*6, self.out_channels*6),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels*6, self.out_channels*6),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels*6, 1),
            nn.Sigmoid()
        )

    
    def forward(self, x, y): # x: ref, y: test
        x1 = self.conv1(x)
        # x1 = self.attention1(x1)
        y1 = self.conv1(y)
        # y1 = self.attention1(y1)

        x2 = self.conv2(x1)
        # x2 = self.attention2(x2)
        y2 = self.conv2(y1)
        # y2 = self.attention2(y2)

        x3 = self.conv3(x2)
        # x3 = self.attention3(x3)
        y3 = self.conv3(y2)
        # y3 = self.attention3(y3)

        x1 = self.mp1(x1)
        y1 = self.mp1(y1)
        x2 = self.mp2(x2)
        y2 = self.mp2(y2)

        cat_x = torch.cat([x1,x2,x3], dim=1)
        cat_y = torch.cat([y1,y2,y3], dim=1)

        cat = torch.cat([cat_x, cat_y], dim=1)
        # GAP
        cat = torch.mean(cat.view(cat.size(0), cat.size(1), -1), dim=2)
        out = self.classifier(cat)
        # print("out:",out.item())
        return out



def signatureNet(**kwargs):
    return model()

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    train_loader = getTrainingData(batch_size = train_batch)
    curr_lr = learning_rate
    total_step = len(train_loader)
    sigNet = signatureNet().to(device)
    # lossFunc = nn.L1Loss().to(device)
    lossFunc = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(sigNet.parameters(), lr=curr_lr)

    for epoch in range(max_epochs):
        for idx, sample in enumerate(train_loader):
            img1, img2, label = sample['img1'], sample['img2'], sample['label']
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            # forward to calculate the values
            result = sigNet(img1, img2)

            label = label.float()
            loss = lossFunc(result, label)

            # backward to calculate the derivative
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (idx+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(epoch+1, max_epochs, idx+1, total_step, loss.item()))
        
        # Decay learning rate
        if (epoch+1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)
    
    torch.save(sigNet, "./results/model_{}".format(max_epochs))

def test(model_idx):
    # Test the model
    test_loader = getTestingData(batch_size = 1)
    model = torch.load("./results/model_{}".format(model_idx))
    model.eval()
    with torch.no_grad():
        TP, TN, FP, FN = 0, 0, 0, 0
        for idx, sample in enumerate(test_loader):
            if (idx) % 100 == 1:
                print(idx)
            img1, img2, label = sample['img1'], sample['img2'], sample['label']
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            out = model(img1, img2)
            predict = True if out > thresh else False

            if predict == False and label == True:
                FN += 1
            elif predict == False and label == False:
                FP += 1
            elif predict == True and label == True:
                TP += 1
            elif predict == True and label == False:
                TN += 1
        
        total_cnt = len(test_loader)
        print(TP,TN,FP,FN)
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        f1 = (2*precision*recall) / (precision+recall)
        print('Accuracy of the model on the test images: {} '.format((TP+TN)/total_cnt))
        print('Recall of the model on the test images: {} '.format(recall))
        print('Precision of the model on the test images: {} '.format(precision))
        print('F1 score of the model on the test images: {} '.format(f1))

if __name__ == '__main__':
    train()
    test(1)