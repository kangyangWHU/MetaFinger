import torch
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# train path
tiny_imagenet_pth = "F:/datasets/tiny-imagenet/train"

cifar10_path = 'F:/datasets/'


cifar10_mean = np.array([0.4914, 0.4822, 0.4465])
cifar10_std = np.array([0.2023, 0.1994, 0.2010])

tinyimagenet_mean = np.array([0.4804, 0.4482, 0.3976])
tinyimagenet_std = np.array([0.2770, 0.2691, 0.2822])

