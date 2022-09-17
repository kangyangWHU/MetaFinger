__all__=['MNIST_Loader', 'Cifar10Loader', 'get_imagenet_val_loader']
from sklearn.model_selection import train_test_split

from torchvision import datasets
import config as flags
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset


class DeNormalize(object):
    '''
        denormalize the tensor to [0,1], usage similar to T.Normalize()
    '''
    def __init__(self, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
        self.mean = -np.array(mean)
        self.std = 1/np.array(std)
    def __call__(self, img):
        '''
        :param img: tensor shape [c,h,w]
        :return: tensor with value between [0,1]
        '''
        img = TF.normalize(img, mean=(0,0,0), std=self.std)
        img = TF.normalize(img, mean=self.mean, std=(1,1,1))
        return img


class MyDataSet(Dataset):

    def __init__(self, datas, labels, transform):
        self.transform = transform
        self.datas = datas
        self.labels = labels
        self.len = len(labels)

    def __getitem__(self, item):

        return self.transform(self.datas[item]), self.labels[item]

    def __len__(self):
        return self.len


class MNIST_Loader(object):

    def __init__(self, batch_size, train_transforms=None, val_test_transforms=None, shuffle=True):
        '''
        :param train_transforms: the transform used on train_mnist set
        :param val_test_transforms: transforms used on valadition set and test set
        '''

        self.mnist_path = flags.mnist_path
        self.batch_size = batch_size

        if train_transforms is None:
            train_transforms = T.Compose([
                T.ToPILImage(),
                T.RandomRotation((-15, 15)),
                T.ToTensor(),
                # T.Normalize(mean=(0.1307,), std=(0.3081,))
            ])

        if val_test_transforms is None:
            val_test_transforms = T.Compose([
                T.ToTensor()
            ])

        trainset = datasets.MNIST(root=self.mnist_path, train=True, download=True)

        # split the train_mnist train_data
        # trainset.data : numpy format
        train_datas, val_datas, train_labels, val_labels = train_test_split(trainset.data, trainset.targets,
                                                                            train_size = 50000, random_state=0, stratify=trainset.targets)

        self.train_loader = DataLoader(MyDataSet(train_datas, train_labels, train_transforms),
                                       batch_size=self.batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

        self.val_loader = DataLoader(MyDataSet(val_datas, val_labels, val_test_transforms),
                                     batch_size=self.batch_size)

        self.test_loader = DataLoader(datasets.MNIST(root=self.mnist_path, train=False, download=True,
                                        transform=val_test_transforms), batch_size=self.batch_size, shuffle=shuffle)


class Cifar100Loader(object):

    def __init__(self, batch_size, train_transforms=None,
                 val_test_transforms=None, num_workers=4, shuffle=True, pin_memory=True):

        self.batch_size = batch_size
        self.cifar100_path = flags.cifar100_path
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        if train_transforms is None:
            train_transforms = T.Compose([
                                    T.ToPILImage(),
                                    T.RandomCrop(32, padding=4),
                                    T.RandomHorizontalFlip(),
                                    T.ToTensor(),
                                ])

        # if test doesn't preprocess, it perform worse
        if val_test_transforms is None:
            val_test_transforms = T.Compose([
                    T.ToTensor()
                ])


        trainset = datasets.CIFAR100(self.cifar100_path, train=True, download=True)

        # trainset.data : numpy format
        train_datas, val_datas, train_labels, val_labels = train_test_split(trainset.data, trainset.targets,
                                                                            train_size=49000, random_state=0, stratify=trainset.targets)

        self.train_loader = DataLoader(MyDataSet(train_datas, train_labels, train_transforms),
                                       batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=self.pin_memory)

        self.val_loader = DataLoader(MyDataSet(val_datas, val_labels, val_test_transforms), num_workers=num_workers,
                                     batch_size=self.batch_size)

        self.test_loader = DataLoader(
                    datasets.CIFAR100(self.cifar100_path, train=False, download=True, transform=val_test_transforms),
                    batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=self.pin_memory)



class Cifar10Loader(object):

    def __init__(self, batch_size, train_transforms=None,
                 val_test_transforms=None, num_workers=4, shuffle=False, pin_memory=False):

        self.batch_size = batch_size
        self.cifar10_path = flags.cifar10_path
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        if train_transforms is None:
            train_transforms = T.Compose([
                                    T.ToPILImage(),
                                    T.RandomCrop(32, padding=4),
                                    T.RandomHorizontalFlip(),
                                    T.ToTensor()
                                ])

        # if test doesn't preprocess, it perform worse
        if val_test_transforms is None:
            val_test_transforms = T.Compose([
                    T.ToTensor(),
                ])


        trainset = datasets.CIFAR10(self.cifar10_path, train=True, download=False)

        # Todo change the transforms of validation set will also change the train set
        # trainset.dataset.transform = train_transforms
        # val_set.dataset.transform = self.val_test_transforms
        # trainset.data : numpy format
        train_datas, val_datas, train_labels, val_labels = train_test_split(trainset.data, trainset.targets,
                                                                            train_size=49000, random_state=0, stratify=trainset.targets)

        self.train_loader = DataLoader(MyDataSet(train_datas, train_labels, train_transforms),
                                       batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=self.pin_memory)

        self.val_loader = DataLoader(MyDataSet(val_datas, val_labels, val_test_transforms),
                                     batch_size=self.batch_size)

        self.test_loader = DataLoader(
                    datasets.CIFAR10(self.cifar10_path, train=False, download=True, transform=val_test_transforms),
                    batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=self.pin_memory)


def get_imagenet_val_loader(path, batch_size, image_size=224, normalize=None,
                            shuffle=False, num_workers=4, pin_memory=False):
    trans = [
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor()
    ]

    if normalize == 'torch':
        trans.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    elif normalize=='tf':
        trans.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    dataset = ImageFolder(path, T.Compose(trans))
    return DataLoader(dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers,pin_memory=pin_memory)
