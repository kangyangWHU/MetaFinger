import os
import sys
sys.path.append("..")

from defense_utils import InputTransformModel
import config as flags
import torch


def load_cifar_model(pth):

    from net.wide_resnet import cifar_wide_resnet
    from net.densenet import densenet_cifar
    from net.vgg import vgg19_bn_cifar10
    from net.inception import inceptionv3_cifar10

    def try_model(model, pth):
        try:
            model.load_state_dict(torch.load(pth)["model"])
        except:
            return None
        return model

    if try_model(cifar_wide_resnet(), pth):
        return try_model(cifar_wide_resnet(), pth)
    elif try_model(vgg19_bn_cifar10(), pth):
        return try_model(vgg19_bn_cifar10(), pth)
    elif try_model(densenet_cifar(), pth):
        return try_model(densenet_cifar(), pth)
    elif try_model(inceptionv3_cifar10(), pth):
        return try_model(inceptionv3_cifar10(), pth)
    else:
        return None

def load_tinyimagenet_model(pth):
    from net.my_resnet import resnet18_tinyimagenet
    from net.vgg import vgg19_bn_tinyimagenet
    def try_model(model, pth):
        try:
            model.load_state_dict(torch.load(pth)["model"])
        except:
            return None
        return model

    if try_model(resnet18_tinyimagenet(), pth):
        return try_model(resnet18_tinyimagenet(), pth)
    elif try_model(vgg19_bn_tinyimagenet(), pth):
        return try_model(vgg19_bn_tinyimagenet(), pth)
    # elif try_model(vgg19_bn_tinyimagenet(), pth):
    #     return try_model(densenet_cifar(), pth)
    # elif try_model(inceptionv3_cifar10(), pth):
    #     return try_model(inceptionv3_cifar10(), pth)
    else:
        return None
def load_wm(path, mean, std):
    checkpoint = torch.load(path)
    x_wm, y_wm = checkpoint["x_wm"], checkpoint["y_wm"]

    # denormalize to [0, 1]
    x_wm *= std.reshape((1, 3, 1, 1))
    x_wm += mean.reshape((1, 3, 1, 1))

    x_wm = torch.tensor(x_wm, dtype=torch.float32)
    y_wm = torch.tensor(y_wm, dtype=torch.long)

    return x_wm, y_wm



def get_dataset(path):

    def helper(path, label):
        train_dataset = []
        for root, dirs, files in os.walk(path):
            if files:
                for name in files:
                    train_dataset.append((os.path.join(root, name), label))
        return train_dataset

    pos_data = helper(os.path.join(path, "positive_models"), 1)
    neg_data = helper(os.path.join(path, "negative_models"), 0)
    return pos_data, neg_data


def model_forward(model_pth, inp, train=True):
    model = load_cifar_model(model_pth)
    model = InputTransformModel(model, normalize=(flags.cifar10_mean, flags.cifar10_std))

    # model = RandomPruning(model, sparsity=0.4).remove()
    model.cuda()
    model.eval()
    out = model.forward(inp)
    return out


def get_acc(model, data_loader):

    model = model.to(flags.device)
    model.eval()
    correct = 0
    for data, label in data_loader:
        data, label = data.to(flags.device,  non_blocking=True), label.to(flags.device, non_blocking=True)

        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(label).sum()

    # Calculate final accuracy for this epsilon
    acc = float(correct) / float(len(data_loader.dataset))
    # Return the accuracy
    return acc*100