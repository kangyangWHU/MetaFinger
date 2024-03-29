import os
import torch
from defense_utils import *
import numpy as np
import config as flags
from torch import nn
import torch
import copy
from scipy import stats
import shutil
from metric_utils import load_cifar_model, get_dataset, load_tinyimagenet_model, load_imagenet_model
from torchvision import datasets
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from torchvision.transforms import functional as TF

class MetaFinger():
    def __init__(self, dataset="cifar10", n_sample=3, alpha=1,  n_query = 100,
                 batch_size = 70,  trans=True, epoch=10, lr=0.01):
        super(MetaFinger, self).__init__()
        self.dataset = dataset
        self.train_path = "weights/{}/train_data".format(dataset)
        self.val_path = "weights/{}/val_data".format(dataset)
        self.test_path = "weights/{}/test_data".format(dataset)
        self.n_sample = n_sample
        self.alpha = alpha
        self.n_query = n_query
        self.batch_size = batch_size
        self.trans = trans
        self.epoch = epoch
        self.lr = lr
        self.init()
        pass

    def init(self):

        self.trans_list = [
            IdentityMap(),
            HFlip(),
            RandShear(shearx=(0, 0.1), sheary=(0, 0.1)),
            GaussianBlur(p=0.7),
            MedianBlur(ksize=3),
            AverageBlur(ksize=3),
            GaussainNoise(0, 0.05),
            UniformNoise(0.05),
        ]

        if self.dataset == "cifar10":
            self.mean = flags.cifar10_mean
            self.std = flags.cifar10_std
            self.trans_list += [RP2(size=(24, 40)), RandTranslate(tx=(0, 5), ty=(0, 5))]
            self.load_model = load_cifar_model
        elif self.dataset == "tinyimagenet":
            self.mean = flags.tinyimagenet_mean
            self.std = flags.tinyimagenet_std
            self.trans_list += [RP2(size=(56, 72)), RandTranslate(tx=(0, 10), ty=(0, 10))]
            self.load_model = load_tinyimagenet_model
        else:
            self.mean = flags.imagenet_mean
            self.std = flags.imagenet_std
            self.trans_list += [RP2(size=(208, 240)), RandTranslate(tx=(0, 10), ty=(0, 10))]
            self.load_model = load_imagenet_model

    def model_forward(self, model_pth, inp):
        model = self.load_model(model_pth)
        model = InputTransformModel(model, normalize=(self.mean, self.std))

        model.cuda()
        model.eval()
        out = model(inp)
        return out

    def get_preds(self, inp, dataset):
        preds_list = []
        for path in dataset:
            preds = self.model_forward(path[0], inp)
            preds_list.append(preds.argmax(1).cpu().numpy())
        preds_list = np.array(preds_list)
        return preds_list

    def epoch_eval(self, inp, pos_dataset, neg_dataset, y=None):


        with torch.no_grad():
            pos_preds = self.get_preds(inp, pos_dataset)
            neg_preds = self.get_preds(inp, neg_dataset)

        if y is None:
            y = stats.mode(pos_preds, axis=0)[0][0]

        pos_acc = (pos_preds==y).sum(1) / len(y)
        neg_acc = (neg_preds==y).sum(1) / len(y)

        return pos_acc, neg_acc, y


    def epoch_train(self, inp, optimizer, pos_dataset, neg_dataset):
        cum_loss = 0.0
        perm_anchor = np.random.permutation(len(pos_dataset))
        for i in perm_anchor:

            # transform inp to enhance its robustness against image transformation

            adv = inp
            if self.trans:
                idx = np.random.randint(0, len(self.trans_list))
                adv = self.trans_list[idx](inp)

            anchor_out = self.model_forward(pos_dataset[i][0], adv)

            loss1 = 0
            for j in np.random.choice(np.arange(len(pos_dataset)), self.n_sample, replace=False):
                pos_out = self.model_forward(pos_dataset[j][0], adv)
                loss1 += nn.KLDivLoss(reduce="batchmean")(anchor_out.log_softmax(1), pos_out.softmax(1))

            loss2 = 0
            for k in np.random.choice(np.arange(len(neg_dataset)), self.n_sample, replace=False):
                neg_out = self.model_forward(neg_dataset[k][0], adv)
                loss2 += nn.KLDivLoss(reduce="batchmean")(anchor_out.log_softmax(1), neg_out.softmax(1))

            # nn.TripletMarginLoss
            loss = loss1/self.n_sample - self.alpha*loss2/self.n_sample

            # probability
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # meta_model.inp.
            inp.data.clamp_(0, 1)

            cum_loss = cum_loss + loss.item()

        return cum_loss

    def screen_queryset(self, inp,  pos_dataset, neg_dataset, FPR=0.0, TPR=1.0):


        with torch.no_grad():
            pos_preds = self.get_preds(inp, pos_dataset)
            neg_preds = self.get_preds(inp,  neg_dataset)

        y = stats.mode(pos_preds, axis=0)[0][0]

        pos_acc = (pos_preds==y).sum(0) / len(pos_dataset)
        neg_acc = (neg_preds==y).sum(0) / len(neg_dataset)

        mask = (pos_acc >= TPR) & (neg_acc <= FPR)

        return inp[mask], y[mask], mask


    def get_inp(self):
        if self.dataset == "cifar10":
            trainset = datasets.CIFAR10(flags.cifar10_path, train=True, download=False)
            idx = np.random.choice(np.arange(30000), size=(self.batch_size,), replace=False)
            noise, labels = trainset.data[idx], np.array(trainset.targets)[idx]

            noise = np.transpose(noise, (0, 3, 1, 2))
            inp = torch.tensor(noise, dtype=torch.float32, requires_grad=True, device="cuda")
            inp.data /= 255

            return inp, labels
        else:
            if self.dataset == "tinyimagenet":
                trainset = datasets.ImageFolder(flags.tiny_imagenet_pth)
            else:
                trainset = datasets.ImageFolder(flags.imagenet12_path)

            idx = np.random.choice(np.arange(30000), size=(self.batch_size, ), replace=False)

            img_list, label_list = [], []
            for index in idx:
                path, label = trainset.imgs[index]
                img = Image.open(path)

                if self.dataset == "imagenet":
                    img = TF.resize(img, 224)
                    img = TF.center_crop(img, 224)

                img = np.array(img)

                if img.shape != (224, 224, 3):
                    continue

                img = np.expand_dims(img, 0)
                img_list.append(img)
                label_list.append(label)

            img_list = np.vstack(img_list)
            label_list = np.array(label_list)
            img_list = np.transpose(img_list, (0, 3, 1, 2))

            inp = torch.tensor(img_list, dtype=torch.float32, requires_grad=True, device="cuda")
            inp.data /= 255

            return inp, label_list

    def gen_queryset(self):
        train_pos, train_neg = get_dataset(self.train_path)
        val_pos, val_neg = get_dataset(self.val_path)
        final_queryset, final_lables = [],[]
        original_set, original_labels = [], []

        for _ in range(100):

            inp, o_labels = self.get_inp()
            original_inp = copy.deepcopy(inp.data)

            optimizer = torch.optim.Adam([inp], lr=self.lr)

            # optimizer = torch.optim.SGD([inp], lr=1e-1) # very bad
            best_neg_acc,best_pos_acc, best_inp = 1, 0, None
            for i in range(self.epoch):

                loss = self.epoch_train(inp, optimizer, train_pos, train_neg)
                print("Epoch:{}, loss:{:.4f}".format(i, loss))

                pos_acc, neg_acc, y = self.epoch_eval(inp.data, train_pos, train_neg)
                print("[Train:] mean pos acc:{:.4f}, mean neg acc:{:.4f}".format(pos_acc.mean(), neg_acc.mean()))
                pos_acc, neg_acc, _ = self.epoch_eval(inp.data, val_pos, val_neg, None)
                print("[Val:  ] mean pos acc:{:.4f}, mean neg acc:{:.4f}".format(pos_acc.mean(), neg_acc.mean()))
                # we only consider the generalization and false positive
                # because the fasle negative is extremely low.
                if neg_acc.mean() <= best_neg_acc and i > 4:
                    best_neg_acc = neg_acc.mean()
                    best_inp =inp.detach().clone()

                torch.cuda.empty_cache()

            queryset, querylabels, query_mask = self.screen_queryset(best_inp.data, val_pos, val_neg)

            print("query length:{}".format(len(queryset)))
            final_queryset.append(queryset.cpu().numpy())
            final_lables += querylabels.tolist()

            original_set.append(original_inp[query_mask].cpu().numpy())
            original_labels += o_labels[query_mask].tolist()

            if len(final_lables) >= self.n_query:
                break

        query_set = np.vstack(final_queryset)[:self.n_query]
        query_labels = np.array(final_lables)[:self.n_query]
        query_set = torch.tensor(query_set, dtype=torch.float32)
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        torch.save(query_set, "query_data/{}_meta_queryset.pth".format(self.dataset))
        torch.save(query_labels, "query_data/{}_meta_querylabels.pth".format(self.dataset))

        # save original query set
        np.save("query_data/{}_original_queryset.npy".format(self.dataset), np.vstack(original_set)[:self.n_query])
        np.save("query_data/{}_original_querylabels.npy".format(self.dataset), np.array(original_labels)[:self.n_query])


    def eval_queryset(self):
        queryset = torch.load("query_data/{}_meta_queryset.pth".format(self.dataset))
        querylabels = torch.load("query_data/{}_meta_querylabels.pth".format(self.dataset))
        queryset = queryset.cuda()

        test_pos, test_neg = get_dataset(self.test_path)

        pos_acc, neg_acc, _ = self.epoch_eval(queryset, test_pos, test_neg, y=querylabels.numpy())
        # print("[Test:] mean pos acc:{:.4f}, mean neg acc:{:.4f}".format(pos_acc.mean(), neg_acc.mean()))

        return pos_acc.mean(), neg_acc.mean()


def gen_cifar_queryset():
    metafinger = MetaFinger(dataset="cifar10", n_sample=3, alpha=1, n_query=100,
                batch_size=70, trans=True, epoch=10, lr=0.01)

    metafinger.gen_queryset()
    pos, neg = metafinger.eval_queryset()
    print("pos :{}, neg:{}".format(pos, neg))


def gen_tinyimagenet_queryset():

    metafinger = MetaFinger(dataset="tinyimagenet", n_sample=3, alpha=1, n_query=100,
                            batch_size=100, trans=True, epoch=10, lr=0.04)

    metafinger.gen_queryset()
    pos, neg = metafinger.eval_queryset()
    print("pos :{}, neg:{}".format(pos, neg))

def gen_imagenet_queryset():

    metafinger = MetaFinger(dataset="imagenet", n_sample=3, alpha=1, n_query=100,
                            batch_size=10, trans=True, epoch=10, lr=0.04)

    metafinger.gen_queryset()
    pos, neg = metafinger.eval_queryset()
    print("pos :{}, neg:{}".format(pos, neg))

import torch
# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构

same_seeds(0)
if __name__ == '__main__':
    # gen_cifar_queryset()
    # gen_tinyimagenet_queryset()
    gen_imagenet_queryset()
    # eval_queryset()