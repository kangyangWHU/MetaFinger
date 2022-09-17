import torch.nn as nn

__all__ = [ 'vgg13_bn_cifar10',  'vgg16_bn_cifar10','vgg19_bn_cifar10',
            'vgg13_bn_cifar100','vgg16_bn_cifar100','vgg19_bn_cifar100']
import mlconfig
class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 4096),
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

        # function to extact the multiple features

    def feature_list(self, x):
        out_list = []

        for i in range(0, 7):
            x = self.features[i](x)
        out_list.append(x)

        for i in range(7, 14):
            x = self.features[i](x)
        out_list.append(x)

        for i in range(14, 28):
            x = self.features[i](x)
        out_list.append(x)

        for i in range(28, 40):
            x = self.features[i](x)
        out_list.append(x)

        for i in range(40, 53):
            x = self.features[i](x)
        out_list.append(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x, out_list

        # function to extact a specific feature

    def intermediate_forward(self, x, layer_index):

        for i in range(0, 7):
            x = self.features[i](x)

        if layer_index == 1:
            for i in range(7, 14):
                x = self.features[i](x)
        elif layer_index == 2:
            for i in range(7, 28):
                x = self.features[i](x)
        elif layer_index == 3:
            for i in range(7, 40):
                x = self.features[i](x)
        elif layer_index == 4:
            for i in range(7, 53):
                x = self.features[i](x)

        return x
        # function to extact the penultimate features

    def penultimate_forward(self, x):
        x = self.features(x)
        features = self.avgpool(x)
        x = features.reshape(features.size(0), -1)
        x = self.classifier(x)
        return x, features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm):

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm))
    return model

#
# def vgg11(pretrained=False, progress=True, **kwargs):
#     """VGG 11-layer nets (configuration "A")
#
#     Args:
#         pretrained (bool): If True, returns a nets pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)

#
# def vgg11_bn(pretrained=False, progress=True, device='cpu', **kwargs):
#     """VGG 11-layer nets (configuration "A") with batch normalization
#
#     Args:
#         pretrained (bool): If True, returns a nets pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg11_bn', 'A', True, pretrained, progress, device, **kwargs)
#
#
# def vgg13(pretrained=False, progress=True, **kwargs):
#     """VGG 13-layer nets (configuration "B")
#
#     Args:
#         pretrained (bool): If True, returns a nets pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)

@mlconfig.register
def vgg13_bn_cifar10(**kwargs):
    return VGG(make_layers(cfgs['B'], batch_norm=True), num_classes=10)

@mlconfig.register
def vgg16_bn_cifar10(**kwargs):
    return VGG(make_layers(cfgs['D'], batch_norm=True), num_classes=10)

@mlconfig.register
def vgg19_bn_cifar10(**kwargs):
    return VGG(make_layers(cfgs['E'], batch_norm=True), num_classes=10)

@mlconfig.register
def vgg19_bn_tinyimagenet(**kwargs):
    return VGG(make_layers(cfgs['E'], batch_norm=True), num_classes=200)


@mlconfig.register
def vgg13_bn_cifar100(**kwargs):
    return VGG(make_layers(cfgs['B'], batch_norm=True), num_classes=100)
@mlconfig.register
def vgg16_bn_cifar100(**kwargs):
    return VGG(make_layers(cfgs['D'], batch_norm=True), num_classes=100)
@mlconfig.register
def vgg19_bn_cifar100(**kwargs):
    return VGG(make_layers(cfgs['E'], batch_norm=True), num_classes=100)



