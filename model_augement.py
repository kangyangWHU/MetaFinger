
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
from collections import OrderedDict
from utils.test_utils import get_acc
from utils.defense_utils import InputTransformModel
from utils.loader_utils import Cifar10Loader
import config as flags
import copy

class RandomPruning():
    """
    The attack randomly prune the weights of a model.
    """
    def __init__(
        self,
        classifier,
        threshold=None,
        sparsity=None,
        prune_last_layer=True,
    ):
        """
        :param classifier: A trained classifier.
        :param threshold: Threshold for weight pruning. If the absolute value of a weight is less than threshold,
                          it will be pruned.
        :param sparsity: Percentage of weights to be pruned for a layer.
        :param prune_last_layer: Whether to prune the last layer of the model.
        """
        super(RandomPruning, self).__init__()

        if threshold is None and sparsity is None:
            raise ValueError("Either 'threshold' or 'sparsity' has to be set.")
        if threshold is not None and sparsity is not None:
            raise ValueError("'threshold' and 'sparsity' are mutually exclusive.")

        self.classifier = classifier
        self.threshold = threshold
        self.sparsity = sparsity
        self.prune_last_layer = prune_last_layer

    def remove(self):

        weights = self.classifier.state_dict()
        # Init an empty dict
        pruned_weights = OrderedDict()
        # Looping through all layers
        for i, (l, w) in enumerate(weights.items()):
            # skip batch norm layers
            if 'bn' in l:
                pruned_weights[l] = w
                continue

            if self.prune_last_layer is False and i == len(weights) - 1:
                pruned_weights[l] = w
                break

            w = w.cpu().numpy()
            if self.threshold is not None:
                w[np.abs(w) <= self.threshold] = 0
            else:
                if len(w.shape) > 0:

                    # prune the weights with small absolute value
                    num_pruned = np.ceil(w.size * self.sparsity).astype('int')
                    idx = np.unravel_index(np.argsort(np.abs(w).ravel()), w.shape)
                    idx = tuple(i[:num_pruned] for i in idx)

                    # random
                    # mask = np.random.random(num_pruned) < 0.5
                    # idx = tuple(i[mask] for i in idx)

                    w[idx] = 0

                    # random prune
                    # mask = np.random.random(w.shape) < self.sparsity
                    # w[mask] = 0

            pruned_weights[l] = torch.from_numpy(w)

        self.classifier.load_state_dict(pruned_weights)

        return self.classifier



class WeightPert():
    """
    The attack consists of a weight pruning attack on a given model.
    """
    def __init__(self, classifier):
        """
        Create a :class:`.WeightShifting` instance.
        Shifts the mean of a single filter in each conv layer.

        :param classifier: A trained classifier.
        """
        super(WeightPert, self).__init__()
        self.classifier = classifier

    def remove(self, alpha=10):

        all_params = list(self.classifier.named_parameters())
        for name, param in all_params:
            if "conv" in name and "weight" in name:
                param.data += torch.normal(torch.zeros_like(param.data), param.data.std(0)/ alpha)

        return self.classifier

if __name__=="__main__":

    import os
    from watermark.fail.utils_meta import load_model

    path = "weights/cifar10/train_data"
    train_dataset = []
    for root, dirs, files in os.walk(path):
        if files:
            for name in files:
                full_path = os.path.join(root, name)
                # if "pruning" in name:
                #     full_path = os.path.join(root, name)
                #     os.remove(full_path)
                model = load_model(full_path)
                copyed_model = copy.deepcopy(model)
                for i in range(2):
                    pruning_name = os.path.join(root, "{}_pruning_{}.pth".format(name, i))
                    # sparsity = np.random.uniform(0.2, 0.4)
                    pruned_model = RandomPruning(copyed_model, sparsity=0.3).remove()
                    # pruned_model = WeightPert(copyed_model).remove(alpha=1)

                    test_model = InputTransformModel(model, normalize=(flags.cifar10_mean, flags.cifar10_std))
                    loader = Cifar10Loader(200, num_workers=0)
                    acc = get_acc(test_model, loader.test_loader)
                    print(acc)

                    test_model = InputTransformModel(copyed_model, normalize=(flags.cifar10_mean, flags.cifar10_std))
                    loader = Cifar10Loader(200, num_workers=0)
                    acc = get_acc(test_model, loader.test_loader)
                    print(acc)

                    # test_model = InputTransformModel(pruned_model, normalize=(flags.cifar10_mean, flags.cifar10_std))
                    # loader = Cifar10Loader(200, num_workers=0)
                    # acc = get_acc(test_model, loader.test_loader)
                    # print(acc)
                    # torch.save({"model":model.state_dict()}, pruning_name)


