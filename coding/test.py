from torch.utils.data import Dataset, DataLoader
from enum import Enum
import numpy as np
import glob
import cv2
import os
import itertools

NUM_F_TRAIN = 12
NUM_G_TRAIN = 24
NUM_FG_TRAIN = NUM_G_TRAIN * NUM_F_TRAIN * 29
NUM_GG_TRAIN = NUM_G_TRAIN * (NUM_G_TRAIN - 1)
TRAIN_G_MAPPING = ['0100','0101','0103','0104','0108','0109','0110','0111','0112','0113','0114','0116']
TRAIN_F_MAPPING = ['0100003','0101004','0103002','0104006','0108008','0109001','0109002','0109003','0109004','0109005','0109006','0109007',
                '0110005','0110009','0111010','0112004','0112010','0113001','0113003','0113005','0113006','0113007','0113009','0114008',
                '0116001','0116002','0116007','0116008','0116009']

# def A(index1):
#     a = {}
#     for index in range(index1):
#         i_F_index = index % 12
#         i_F = (index // 12) % 29
#         i_G_index = (index // (12 * 29)) % 24
#         image_F = TRAIN_F_MAPPING[i_F] + '_' + str(i_F_index+1) + '.png'
#         image_G = TRAIN_F_MAPPING[i_F][-3:] + '_' + str(i_G_index+1) + '.png'
#         print(image_F,image_G)


# # A(29*12*24)
# dataPath = 'dataset/sigComp2011/testingSet/'
# for i in range(11, 21):
#     index = '%03d' % i
#     path1 = dataPath + 'Ref(115)/{}/*.PNG'.format(index)                
#     path2_F = dataPath + 'Questioned(487)/{}/*.PNG'.format(index)
#     path2_G = dataPath + 'Questioned(487)/{}/*.png'.format(index)
#     len_ref = len(glob.glob(pathname=path1))
#     len_F = len(glob.glob(pathname=path2_F))
#     len_G = len(glob.glob(pathname=path2_G))
#     len_question = len_F + len_G
#     num = len_ref * len_question
#     print(num, end=',')

# TEST_MAPPING = [576,576,460,564,576,576,649,576,528,470]
# import os
# num1 = 1
# files = os.listdir('./')
# for f in files:
#     new_name = '{}.png'.format(num1)
#     os.rename(f, new_name)
#     num1 += 1

a=[1,2,3]
b=[4,5,6]
a+=b
print(a)

