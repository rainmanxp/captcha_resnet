# -*- coding: UTF-8 -*-
import os,time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from torchvision import models

import sys
import importlib

import my_dataset
import one_hot_encoding
#importlib.reload(my_dataset)


# 验证码中的字符
# string.digits + string.ascii_uppercase
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

ALL_CHAR_SET = NUMBER + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4

CLASS_NUM = MAX_CAPTCHA * ALL_CHAR_SET_LEN

# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

TRAIN_DATASET_PATH = 'dataset' + os.path.sep + 'train'
TEST_DATASET_PATH = 'dataset' + os.path.sep + 'test'
PREDICT_DATASET_PATH = 'dataset' + os.path.sep + 'predict'


RANDOM_SEED = 1
LEARNING_RATE = 0.0002
BATCH_SIZE = 64
NUM_EPOCH = 1

# Architecture
NUM_FEATURES = 160 * 60

# Other
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GRAYSCALE = True

RESNET_DICT = {
    'net' : {
        'resnet18' : models.resnet18,
        'resnet34' : models.resnet34,
        'resnet50' : models.resnet50,
        'resnet101' : models.resnet101,
    },
    'init_weight':  {
        'resnet18' : models.ResNet18_Weights.DEFAULT,
        'resnet34' : models.ResNet34_Weights.DEFAULT,
        'resnet50' : models.ResNet50_Weights.DEFAULT,
        'resnet101' : models.ResNet101_Weights.DEFAULT,
    }
}

SAVE_MODEL_NAME = 'captcha_reg.model'