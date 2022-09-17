import os
from metric_utils import load_cifar_model, load_tinyimagenet_model, get_acc
from defense_utils import InputTransformModel
import config as flags
from loader_utils import Cifar10Loader
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from PIL import Image

class Eval():
    def __init__(self, dataset="cifar10"):
        super(Eval, self).__init__()
        self.dataset = dataset

        self.init()

    def init(self):

        if self.dataset == "cifar10":
            self.mean = flags.cifar10_mean
            self.std = flags.cifar10_std
            self.load_model = load_cifar_model
            self.loader = Cifar10Loader(200, num_workers=0, shuffle=True).test_loader
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

    def eval_queryset(self, root_path, queryset, querylabels):

        queryset = queryset.cuda()
        querylabels = querylabels.cuda()
        for root, dirs, files in os.walk(root_path):
            if files is None:
                continue

            mean_acc = 0
            for i, filename in enumerate(files):
                path = os.path.join(root, filename)
                model = InputTransformModel(self.load_model(path), normalize=(self.mean, self.std))
                model = model.cuda()
                model.eval()
                success = (model(queryset).argmax(1) == querylabels).sum()
                acc = success / len(querylabels)
                mean_acc += acc
                # print("name:{}, acc:{}".format(filename, acc))

                if (i+1) % 3 == 0:
                    print("name:{}, acc:{}".format(filename, mean_acc/3))
                    print()
                    mean_acc = 0


    def eval_clean(self, root_path):


        for root, dirs, files in os.walk(root_path):
            if files is None:
                continue

            mean_acc = 0
            for i, filename in enumerate(files):
                path = os.path.join(root, filename)
                model = InputTransformModel(self.load_model(path), normalize=(self.mean, self.std))
                model = model.cuda()
                model.eval()

                mean_acc += get_acc(model, self.loader)
                # print("name:{}, acc:{}".format(filename, acc))

                if (i+1) % 3 == 0:
                    print("name:{}, acc:{}".format(root, mean_acc/3))
                    print()
                    mean_acc = 0


if __name__=="__main__":

    # eval_clean("weights/test_data/negative_models")
    # IPGuard
    # queryset = torch.load("query_data/tinyimagenet_ipguard_queryset.pth")
    # querylabels = torch.load("query_data/tinyimagenet_ipguard_querylabels.pth")
    # Eval("tinyimagenet").eval_queryset("weights/tinyimagenet/test_data/positive_models", queryset, querylabels)

    #
    dataset = "cifar10"
    queryset = torch.load("query_data/{}_meta_queryset.pth".format(dataset))
    querylabels = torch.load("query_data/{}_meta_querylabels.pth".format(dataset))
    Eval(dataset).eval_queryset("weights/{}/test_data/positive_models".format(dataset), queryset, querylabels)
    Eval(dataset).eval_queryset("weights/{}/test_data/negative_models".format(dataset), queryset, querylabels)


    # print("="*50)
    # Meta Classifier
    # queryset = torch.load("query_data/tinyimagenet_meta_queryset.pth")
    # querylabels = torch.load("query_data/tinyimagenet_meta_querylabels.pth")
    # Eval("tinyimagenet").eval_queryset("weights/tinyimagenet/test_data/negative_models", queryset, querylabels)

    # Adi Scratch
    # queryset, querylabels = load_wm_cifar("weights/adi_scratch.pth")
    # eval_queryset("weights/test_data/negative_models", queryset, querylabels)

    # Adi Pretrained
    # queryset, querylabels = load_wm_cifar("weights/adi_pretrained.pth")
    # eval_queryset("weights/test_data/negative_models", queryset, querylabels)

    # Jia Pretrained
    # queryset, querylabels = load_wm_cifar("weights/jia.pth")
    # eval_queryset("weights/test_data/negative_models", queryset, querylabels)