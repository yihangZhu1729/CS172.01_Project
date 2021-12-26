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
max_epochs = 5
train_batch = 8
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

        self.attention = CBAM(self.out_channels)

    
    def forward(self, x):           # 1 1000 600
        x1 = self.conv1(x)           # 32 500 300
        x1 = self.attention(x1)       

        x2 = self.conv2(x1) 
        x2 = self.attention(x2)

        x3 = self.conv3(x2)
        x3 = self.attention(x3)

        # global average pooling
        x1 = torch.mean(x1.view(x1.size(0), x1.size(1), -1), dim=2)
        x2 = torch.mean(x2.view(x2.size(0), x2.size(1), -1), dim=2)
        x3 = torch.mean(x3.view(x3.size(0), x3.size(1), -1), dim=2)

        x_out = torch.cat([x1,x2,x3], dim =1)        

        return x_out

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
    lossFunc = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(sigNet.parameters(), lr=curr_lr)

    for epoch in range(max_epochs):
        for idx, sample in enumerate(train_loader):
            img1, img2, label = sample['img1'], sample['img2'], sample['label']
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            print("idx", idx)
            # forward to calculate the values
            out1 = sigNet(img1)
            out2 = sigNet(img2)

            similarity = torch.cosine_similarity(out1, out2)
            result = torch.mul(similarity >= thresh,1).float().requires_grad_()
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
    
    # torch.save(depthNet, "./results/model_{}".format(max_epochs))

def test(model_idx):
    # Test the model
    test_loader = getTestingData(batch_size = 2)
    model = torch.load("./results/model_{}".format(model_idx))
    model.eval()
    with torch.no_grad():
        total_REL = 0
        total_Mlog10E = 0
        for idx, sample in enumerate(test_loader):
            rgb, depth = sample['rgb'], sample['depth']
            rgb = rgb.to(device)
            depth = depth.numpy()[0][0]
            output = model(rgb)
            output = output.data.cpu().numpy()[0][0] + 0.0001

            REL = np.mean(np.abs(depth - output) / depth)
            Mlog10E = np.mean(np.abs(np.log10(depth)-np.log10(output)))

            total_REL += REL
            total_Mlog10E += Mlog10E

        test_len = len(test_loader)
        print('Average REL accuracy of the model on the test images: {} '.format(total_REL / test_len))
        print('Average log1oE accuracy of the model on the test images: {} '.format(total_Mlog10E / test_len))

if __name__ == '__main__':
    train()