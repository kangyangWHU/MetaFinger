from defense_utils import *
from loader_utils import Cifar10Loader
import torch
from metric_utils import load_cifar_model, load_tinyimagenet_model, get_acc
import config as flags
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from PIL import Image

class InputTrans():
    def __init__(self, dataset="cifar10"):
        super(InputTrans, self).__init__()
        self.dataset = dataset

        self.init()

    def init(self):

        self.trans_list = [
            IdentityMap(),
            GaussainNoise(0, 0.05),
            UniformNoise(0.05),
            GaussianBlur(),
            MedianBlur(ksize=3),
            AverageBlur(ksize=3),
        ]
        if self.dataset == "cifar10":
            self.mean = flags.cifar10_mean
            self.std = flags.cifar10_std
            self.load_model = load_cifar_model
            self.loader = Cifar10Loader(200, num_workers=0, shuffle=True).test_loader
            self.trans_list.append(RP2(size=(24, 40)))
        else:
            self.mean = flags.tinyimagenet_mean
            self.std = flags.tinyimagenet_std
            self.load_model = load_tinyimagenet_model

            # keep aline with
            transform = T.Compose([
                T.Resize(64 + 16, interpolation=Image.BILINEAR),
                T.CenterCrop(64),
                T.ToTensor(),
            ])

            self.loader = DataLoader(ImageFolder("F:/datasets/tiny-imagenet/val",  transform=transform),
                                     batch_size=200, shuffle=True, num_workers=0)
            self.trans_list.append(RP2(size=(58, 72)))

    def get_test_data_acc(self, model_pth):


        model = self.load_model(model_pth)
        for T in self.trans_list:
            trans_model = InputTransformModel(model, normalize=(self.mean, self.std), input_trans=T)

            acc = 0
            for i in range(3):
                acc += get_acc(trans_model, self.loader)
            print("transforms:{}, acc:{:.4f}".format(T, acc/3))


    def get_queryset_acc(self, model_pth, queryset, querylabels):

        queryset = queryset.cuda()
        querylabels = querylabels.cuda()

        model = self.load_model(model_pth)
        model = InputTransformModel(model, normalize=(self.mean, self.std))
        model.cuda()
        model.eval()

        for T in self.trans_list:

            # average over three times
            acc = 0
            for _ in range(3):
                preds = []
                for i in range(0,queryset.shape[0],2):
                    data = T(queryset[i:i+2])
                    logits = model(data)
                    preds.append(logits.argmax(1))

                acc += (torch.hstack(preds) == querylabels).sum()/ len(querylabels)

            print("transforms:{}, acc:{:.2f}".format(T, (acc/3)*100))


if __name__=="__main__":
    # pass
    # clean acc
    # trans = InputTrans("tinyimagenet")
    # trans.get_test_data_acc("weights/tinyimagenet/source_model.pth")

    # ipguard
    # querylabels  = torch.load("query_data/tinyimagenet_ipguard_querylabels.pth")
    # queryset = torch.load("query_data/tinyimagenet_ipguard_queryset.pth")
    # trans = InputTrans("tinyimagenet")
    # trans.get_queryset_acc("weights/tinyimagenet/source_model.pth", queryset, querylabels)

    # ipguard
    dataset = "tinyimagenet"
    querylabels  = torch.load("query_data/{}_meta_querylabels.pth".format(dataset))
    queryset = torch.load("query_data/{}_meta_queryset.pth".format(dataset))
    trans = InputTrans(dataset)
    trans.get_queryset_acc("weights/{}/source_model.pth".format(dataset), queryset, querylabels)

    # ours
    # querylabels  = torch.load("query_data/tinyimagenet_meta_querylabels.pth")
    # queryset = torch.load("query_data/tinyimagenet_meta_queryset.pth")
    # trans = InputTrans("tinyimagenet")
    # trans.get_queryset_acc("weights/tinyimagenet/source_model.pth", queryset, querylabels)

    # adi
    # path = "weights/tinyimagenet/adi_scratch.pth"
    # queryset,  querylabels = load_wm(path, flags.tinyimagenet_mean, flags.tinyimagenet_std)
    # trans = InputTrans("tinyimagenet")
    # trans.get_queryset_acc(path, queryset, querylabels)

    # path = "weights/tinyimagenet/adi_pretrained.pth"
    # queryset,  querylabels = load_wm(path, flags.tinyimagenet_mean, flags.tinyimagenet_std)
    # trans = InputTrans("tinyimagenet")
    # trans.get_queryset_acc(path, queryset, querylabels)

    # path = "weights/tinyimagenet/jia.pth"
    # queryset,  querylabels = load_wm(path, flags.tinyimagenet_mean, flags.tinyimagenet_std)
    # trans = InputTrans("tinyimagenet")
    # trans.get_queryset_acc(path, queryset, querylabels)


    # path = "weights/jia.pth"
    # queryset,  querylabels = load_wm(path)
    # input_trans(path, queryset, querylabels)
