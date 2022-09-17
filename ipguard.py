import sys
sys.path.append("..")

import config as flags
from defense_utils import InputTransformModel
from loader_utils import Cifar10Loader
import torch
import numpy as np
from net.wide_resnet import cifar_wide_resnet
from metric_utils import load_cifar_model, load_tinyimagenet_model
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

def gen_queryset():

    pth = "weights/train_data/positive_models/source_models/best.pth"
    k = 10
    number = 100
    n_limits = 1000
    lr = 0.001

    loader = Cifar10Loader(200, num_workers=0, shuffle=True)
    model = cifar_wide_resnet()
    model.load_state_dict(torch.load(pth)["model"])
    model = InputTransformModel(model, normalize=(flags.cifar10_mean, flags.cifar10_std))
    model = model.cuda()
    model.eval()

    query_set = []
    query_labels = []
    for data, labels in loader.train_loader:

        data, labels = data.cuda(), labels.cuda()
        logits = model(data)
        correct_mask = (logits.argmax(1) == labels)
        data = data[correct_mask]
        original_labels = labels[correct_mask]
        target_labels = logits.argmin(1)[correct_mask]


        ## generate pert
        for img, o_label, t_label in zip(data, original_labels, target_labels):

            img = img.unsqueeze(0)
            img.requires_grad = True
            optimizer = torch.optim.Adam([img], lr=lr)
            for iter in range(n_limits):

                logits = model(img).squeeze()

                other_max = 0
                for value, idx in zip(logits.topk(3).values, logits.topk(3).indices):
                    if idx not in [o_label, t_label]:
                        other_max = value
                        break
                loss = torch.relu(logits[o_label] - logits[t_label] + k) + torch.relu(other_max - logits[o_label])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                img.data.clamp_(0, 1)

                if loss == 0:
                    query_set.append(img.detach().cpu().numpy())

                    label = model(img).argmax(1)
                    query_labels.append(label.item())
                    break

            if len(query_set) == number:

                query_set = np.vstack(query_set)
                query_set = torch.tensor(query_set, dtype=torch.float32)
                query_labels = torch.tensor(query_labels, dtype=torch.long)
                torch.save(query_set, "query_data/ipguard_queryset.pth")
                torch.save(query_labels, "query_data/ipguard_querylabels.pth")
                exit()


class IPGuard():
    def __init__(self, pth, dataset="cifar10", lr=0.001, k=10, n_query=100, n_limits=1000):
        super(IPGuard, self).__init__()
        self.pth = pth
        self.dataset = dataset
        self.lr = lr
        self.k = k
        self.n_query = n_query
        self.n_limits = n_limits

        self.init()

    def init(self):

        if self.dataset == "cifar10":
            self.mean = flags.cifar10_mean
            self.std = flags.cifar10_std
            self.load_model = load_cifar_model
            self.loader = Cifar10Loader(200, num_workers=0, shuffle=True).train_loader

        else:
            self.mean = flags.tinyimagenet_mean
            self.std = flags.tinyimagenet_std
            self.load_model = load_tinyimagenet_model
            self.loader = DataLoader(ImageFolder("F:/datasets/tiny-imagenet/train",  transform=T.ToTensor()),
                                     batch_size=200, shuffle=True, num_workers=0)

    def gen_queryset(self):

        model = self.load_model(self.pth)
        model = InputTransformModel(model, normalize=(self.mean, self.std))
        model = model.cuda()
        model.eval()

        query_set = []
        query_labels = []
        for data, labels in self.loader:

            data, labels = data.cuda(), labels.cuda()
            logits = model(data)
            correct_mask = (logits.argmax(1) == labels)
            data = data[correct_mask]
            original_labels = labels[correct_mask]
            target_labels = logits.argmin(1)[correct_mask]

            ## generate pert
            for img, o_label, t_label in zip(data, original_labels, target_labels):

                img = img.unsqueeze(0)
                img.requires_grad = True
                optimizer = torch.optim.Adam([img], lr=self.lr)
                for iter in range(self.n_limits):

                    logits = model(img).squeeze()

                    other_max = 0
                    for value, idx in zip(logits.topk(3).values, logits.topk(3).indices):
                        if idx not in [o_label, t_label]:
                            other_max = value
                            break
                    loss = torch.relu(logits[o_label] - logits[t_label] + self.k) + torch.relu(other_max - logits[o_label])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    img.data.clamp_(0, 1)

                    if loss == 0:
                        query_set.append(img.detach().cpu().numpy())

                        label = model(img).argmax(1)
                        query_labels.append(label.item())
                        break

                if len(query_set) == self.n_query:
                    query_set = np.vstack(query_set)
                    query_set = torch.tensor(query_set, dtype=torch.float32)
                    query_labels = torch.tensor(query_labels, dtype=torch.long)
                    torch.save(query_set, "query_data/{}_ipguard_queryset.pth".format(self.dataset))
                    torch.save(query_labels, "query_data/{}_ipguard_querylabels.pth".format(self.dataset))
                    exit()


if __name__=="__main__":

    # pth = ""
    # ipguard = IPGuard(pth, dataset="cifar10", lr=0.001, k=10, n_query=100, n_limits=1000)
    # ipguard.gen_queryset()

    pth = "weights/tinyimagenet/source.pth"
    ipguard = IPGuard(pth, dataset="tinyimagenet", lr=0.001, k=10, n_query=100, n_limits=1000)
    ipguard.gen_queryset()