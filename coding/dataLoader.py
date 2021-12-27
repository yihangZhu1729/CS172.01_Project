from torch.utils.data import Dataset, DataLoader
from enum import Enum
import numpy as np
import glob
import cv2
import os
import itertools
import torch
import random

NUM_F_TRAIN = 12
NUM_G_TRAIN = 24
NUM_FG_TRAIN = NUM_G_TRAIN * NUM_F_TRAIN * 20
NUM_GG_TRAIN = NUM_G_TRAIN * NUM_G_TRAIN  * 10
TRAIN_G_MAPPING = ['0100','0101','0103','0104','0108','0109','0110','0111','0112','0113','0114','0116']
TRAIN_F_MAPPING = ['0100003','0101004','0103002','0104006','0108008','0109005',
                '0110005','0111010','0112004','0112010','0113001','0113003','0113006','0113007','0113009','0114008',
                '0116001','0116002','0116007','0116009']
TEST_MAPPING = [576,1152,1612,2176,2752,3328,3977,4553,5081,5551]


Datatype = Enum('Datatype', ('TRAIN', 'TEST'))

class signatureDataset(Dataset):
    """ SigComp11-NFI signature dataset """

    def __init__(self, dataPath, dataType, transform=None):
        self.dataPath = dataPath
        self.dataType = dataType

        if dataType is Datatype.TRAIN:
            # path_forgery = self.dataPath + '/Offline Forgeries/*.png'
            path2_genuine = self.dataPath + '/Offline Genuine/*.png'
            # 12 png for 1 fake signature, 24 png for 1 genuine signature
            # num = len(glob.glob(pathname=path2_genuine)) / NUM_G_TRAIN # 10
            self.imgPairs = NUM_GG_TRAIN + NUM_FG_TRAIN
        
        if dataType is Datatype.TEST:
            self.imgPairs = 5551

            image_list = []
            for i in range(11,21):
                files_Q = os.listdir('dataset/sigComp2011/testingSet/Questioned(487)/{}'.format('%03d'%i))
                files_R = os.listdir('dataset/sigComp2011/testingSet/Ref(115)/{}'.format('%03d'%i))
                i_list = list(itertools.product(files_Q,files_R))
                image_list += i_list
            self.image_list = image_list
    
    def __getitem__(self, index):
        image_1 = None
        image_2 = None
        if self.dataType is Datatype.TRAIN:
            if index < NUM_FG_TRAIN:
                i_F_index = index % 12
                i_F = (index // 12) % 20
                i_G_index = (index // (12 * 20)) % 24
                image_1 = r'dataset/sigComp2011/trainingSet/Offline Forgeries/' + TRAIN_F_MAPPING[i_F] + '_' + str(i_F_index+1) + '.png'
                image_2 = r'dataset/sigComp2011/trainingSet/Offline Genuine/' + TRAIN_F_MAPPING[i_F][-3:] + '_' + str(i_G_index+1) + '.png'
                label = False
            else:
                index -= NUM_FG_TRAIN
                i_person = index // (NUM_G_TRAIN * NUM_G_TRAIN )
                i_image = index // 20
                i_list = list(itertools.product([i for i in range(1,25)],[i for i in range(1,25)]))
                image_1 = r'dataset/sigComp2011/trainingSet/Offline Genuine/' + '%03d' % (i_person+1) + '_' + str(i_list[i_image][0]) + '.png'
                image_2 = r'dataset/sigComp2011/trainingSet/Offline Genuine/' + '%03d' % (i_person+1) + '_' + str(i_list[i_image][1]) + '.png'
                label = True
        else:
            num = 0
            for i in range(10):
                if index < TEST_MAPPING[i]:
                    num = 11 + i 
                    break
            image_1 = 'dataset/sigComp2011/testingSet/Questioned(487)/{}/{}'.format('%03d'%num,self.image_list[index][0])
            image_2 = 'dataset/sigComp2011/testingSet/Ref(115)/{}/{}'.format('%03d'%num,self.image_list[index][1])
            label = 1. if 'G' in self.image_list[index][0] else 0.
        
        img_1 = cv2.resize((cv2.imread(image_1, 0)).astype(np.float32), (320, 160))
        img_2 = cv2.resize((cv2.imread(image_2, 0)).astype(np.float32), (320, 160))
        # when training, by a random factor, we inverse the image color
        if self.dataType is Datatype.TRAIN:
            if random.random() < 0.5:
                img_1 = cv2.bitwise_not(img_1)
            else:
                img_2 = cv2.bitwise_not(img_2)

        img_1 = np.array([img_1])
        img_2 = np.array([img_2])

        sample = {'img1': img_1, 'img2':img_2, 'label':label}
        return sample
    def __len__(self):
        return self.imgPairs


def getTrainingData(batch_size = 64):
    dataSet = signatureDataset(dataPath = r'./dataset/sigComp2011/trainingSet', dataType = Datatype.TRAIN)
    data_loader = DataLoader(dataSet, batch_size, shuffle = True, num_workers = 0, pin_memory = False)
    return data_loader

def getTestingData(batch_size = 1):
    dataSet = signatureDataset(dataPath = r'./dataset/sigComp2011/testingSet', dataType = Datatype.TEST)
    data_loader = DataLoader(dataSet, batch_size, shuffle = False, num_workers = 0, pin_memory = False)
    return data_loader