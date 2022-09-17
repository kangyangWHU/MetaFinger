__all__ = ["RP","RP2", "IdentityMap","Resize","RandResize" ,"BitDepthReduce", "GaussianBlur", "AverageBlur", "MedianBlur",
           "MedianBlurEven", "AverageBlurEven",
           "ShiftChannel", "DropPixel", "UniformNoise","GaussainNoise", "RandRotate", "HFlip",
           "VFlip", "RandShear", "GrayScale", "RandTranslate", "DWT_Denoise",
           "JPEG", "NonLocalMean","GaussianNoise_AUG","CutOut_AUG", "CoarseDropout_AUG",
           "CropResizePad_AUG",
           "InputTransformModel", "InputTransformWraper"]

import PIL
import PIL.Image
from io import BytesIO
import sys
sys.path.append("..")
from pytorch_wavelets import DWT,IDWT
import torch
from torch import nn
import imgaug.augmenters as iaa
import kornia
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from typing import Tuple, List


class IdentityMap():
    def __init__(self, p=1):
        self.p = p

    def __call__(self, x):
        return x


#  differentiable
class RP():
    def __init__(self, max_size=331, value=0.5, p=1):
        self.max_size = max_size
        self.value = value
        self.p = p

    def rp(self, x):
        rnd = np.random.randint(x.shape[-1], self.max_size)
        x = kornia.resize(x, size=(rnd, rnd))

        h_rem = self.max_size - rnd
        w_rem = self.max_size - rnd

        pad_left = np.random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        pad_top = np.random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top

        x = torch.nn.functional.pad(x, [pad_left, pad_right,pad_top, pad_bottom], mode='constant', value=self.value)
        return x

    def __call__(self, x:torch.Tensor):

        # perform transform
        if np.random.rand() < self.p:
            return self.rp(x)
        else:
            return x

#  differentiable
class RP2():
    def __init__(self, size=(24, 40), value=0.5, p=1):
        self.size = size
        self.value = value
        self.p = p

    def rp(self, x):
        rnd = np.random.randint(self.size[0], self.size[1])
        x = kornia.resize(x, size=(rnd, rnd))

        h_rem = self.size[1] - rnd
        w_rem = self.size[1] - rnd

        pad_left = np.random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        pad_top = np.random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top

        x = torch.nn.functional.pad(x, [pad_left, pad_right,pad_top, pad_bottom], mode='constant', value=self.value)
        return x

    def __call__(self, x:torch.Tensor):

        # perform transform
        if np.random.rand() < self.p:
            return self.rp(x)
        else:
            return x


class Resize():
    def __init__(self, size=224, p=1, mode="bilinear"):
        super(Resize, self).__init__()
        self.size = size
        self.p = p
        self.mode = mode
        if self.mode == "bilinear":
            self.align_corners=True
        elif self.mode == "nearest":
            self.align_corners = None

    def __call__(self, x):
        if np.random.rand() < self.p:
            x = torch.nn.functional.interpolate(x, size=self.size
                                                , mode=self.mode
                                                , align_corners = self.align_corners
                                                )
        return x

class RandResize():
    def __init__(self, size=(224, 225), p=1, mode="bilinear"):
        super(RandResize, self).__init__()
        self.size = size
        self.p = p
        self.mode = mode
        if self.mode == "bilinear":
            self.align_corners=True
        elif self.mode == "nearest":
            self.align_corners = None

    def __call__(self, x):
        if np.random.rand() < self.p:
            size = np.random.randint(self.size[0], self.size[1])
            x = torch.nn.functional.interpolate(x, size=size
                                                , mode=self.mode
                                                , align_corners = self.align_corners
                                                )
        return x

class GaussianBlur():

    def __init__(self,  ksize=3, sigma=0.8, p=1):
        super(GaussianBlur, self).__init__()
        self.ksize = ksize
        self.sigma = sigma
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            x = kornia.gaussian_blur2d(x,  (self.ksize, self.ksize), (self.sigma, self.sigma))
        return x

def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalizes both derivative and smoothing kernel.
    """
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))


def filter2D(input: torch.Tensor, kernel: torch.Tensor,
             normalized: bool = False) -> torch.Tensor:

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(0).to(input.device).to(input.dtype)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    return F.conv2d(input, tmp_kernel.expand(c, -1, -1, -1), groups=c, padding=0, stride=1)


def get_box_kernel2d(kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Utility function that returns a box filter."""
    kx: float = float(kernel_size[0])
    ky: float = float(kernel_size[1])
    scale: torch.Tensor = torch.tensor(1.) / torch.tensor([kx * ky])
    tmp_kernel: torch.Tensor = torch.ones(1, kernel_size[0], kernel_size[1])
    return scale.to(tmp_kernel.dtype) * tmp_kernel

class AverageBlurEven():

    def __init__(self,  ksize, p=1):
        if ksize %2 !=0:
            raise TypeError("ksize must be even")

        kernel = get_box_kernel2d((ksize, ksize))
        self.kernel = kernel
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            b, c, h, w = x.shape
            blur_x = F.conv2d(x, self.kernel.expand(c, -1, -1, -1), groups=c, padding=0, stride=1)

            res = x.clone()
            res[:, :, 1:, 1:] = blur_x
            return res
        return x



class AverageBlur():
    def __init__(self, ksize, p=1):
        super(AverageBlur, self).__init__()
        self.p = p
        self.ksize = ksize
    def __call__(self, x):
        if np.random.rand() < self.p:
            x = kornia.box_blur(x,  (self.ksize, self.ksize))
        return x


class MedianBlur():
    def __init__(self, ksize, p=1):
        super(MedianBlur, self).__init__()
        self.p = p
        self.ksize = ksize
    def __call__(self, x):
        if np.random.rand() < self.p:
            x = kornia.median_blur(x, (self.ksize, self.ksize))
        return x

def get_binary_kernel2d(window_size: Tuple[int, int]) -> torch.Tensor:
    r"""Creates a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])

class MedianBlurEven():

    def __init__(self, ksize, p=1):
        self.kernel = get_binary_kernel2d((ksize, ksize))
        self.p = p

    def __blur__(self, input):

        # prepare kernel
        b, c, h, w = input.shape
        kernel = self.kernel.to(input.device).to(input.dtype)
        # map the local window to single vector
        features = F.conv2d(
            input.reshape(b * c, 1, h, w), kernel, stride=1)
        features = features.view(b, c, -1, h-1, w-1)  # BxCx(K_h * K_w)xHxW

        # compute the median along the feature axis
        median = torch.median(features, dim=2)[0]
        return median

    def __call__(self, x):
        if np.random.rand() < self.p:
            blur_x = self.__blur__(x)
            res = x.clone()
            res[:,:,1:,1:] = blur_x
            return res
        return x



class ShiftChannel():
    def __init__(self, p=1):
        self.p = p

    def __call__(self, x):

        if np.random.rand() < self.p:
            x = x.flip(1)
        return x

class DropPixel():
    def __init__(self, drop_rate=0.1, p=1):
        super(DropPixel, self).__init__()
        self.p = p
        self.drop_rate = drop_rate

    def __call__(self, x):

        if np.random.rand() < self.p:
            x_copy = x.clone()
            drop_mask = torch.rand(size=x_copy.shape) < self.drop_rate
            x_copy[drop_mask] = x_copy.median()
            return x_copy
        return x


class UniformNoise():
    def __init__(self, epsilon,  p=1):
        '''
        :param epsilon: value between [0,1]
        '''
        self.epsilon = epsilon
        self.p = p
    def __call__(self, x):
        if np.random.rand() < self.p:
            noise = np.random.uniform(-self.epsilon, self.epsilon, size=x.shape)
            noise = torch.from_numpy(noise)
            noise = noise.to(dtype=x.dtype, device=x.device)
            return torch.clamp(x+noise, 0, 1)
        return x



class GaussainNoise():
    def __init__(self, mu, sigma,  p=1):

        self.mu = mu
        self.sigma = sigma
        self.p = p
    def __call__(self, x):
        if np.random.rand() < self.p:
            noise = np.random.normal(self.mu, self.sigma, size=x.shape)
            noise = torch.from_numpy(noise)
            noise = noise.to(dtype=x.dtype, device=x.device)
            return torch.clamp(x+noise, 0, 1)
        return x

class RandRotate():
    def __init__(self, angle=(0,15), p=1):
        '''
        :param angle: always be positive, it will be random select a direction to rotate
        :param p:
        '''
        self.angle = angle
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            angle = torch.randint(self.angle[0], self.angle[1], ())

            if np.random.rand() < 0.5:
                angle = -angle
            x = kornia.rotate(x, angle=angle.to(x.device, dtype=torch.float32))
        return x


class HFlip():
    def __init__(self, p=1):
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            x = kornia.hflip(x)
        return x


class VFlip():
    def __init__(self, p=1):
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            x = kornia.vflip(x)
        return x


class RandShear():
    def __init__(self,shearx=(0,0.1), sheary=(0,0.1), p=1):
        '''
        :param shearx: always be positive, it will be random select a direction to rotate
        :param sheary: same as shearx
        :param p:
        '''
        self.p = p
        self.shearx = shearx
        self.sheary = sheary


    def __call__(self, x):
        if np.random.rand() < self.p:
            shearx = np.random.uniform(self.shearx[0], self.shearx[1])
            sheary = np.random.uniform(self.sheary[0], self.sheary[1])

            if np.random.rand() < 0.5:
                shearx = -shearx

            if np.random.rand() < 0.5:
                sheary = - sheary

            shear = torch.tensor([shearx, sheary]).unsqueeze(0)
            shear = shear.repeat((x.shape[0],1))
            x = kornia.shear(x, shear.to(device=x.device, dtype=x.dtype))
        return x

class GrayScale():
    def __init__(self, p=1):
        self.p = p
    def __call__(self, x):
        if np.random.rand() < self.p:
            x = kornia.kornia.rgb_to_grayscale(x)
        return x


class RandTranslate():

    def __init__(self, tx=(0, 20), ty=(0, 20), p=1):
        self.tx = tx
        self.ty = ty
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            tx = np.random.randint(self.tx[0], self.tx[1])
            ty = np.random.randint(self.ty[0], self.ty[1])

            if np.random.rand() < 0.5:
                tx = -tx
            if np.random.rand() < 0.5:
                ty = -ty

            translate = torch.tensor([tx, ty], device=x.device, dtype=x.dtype).unsqueeze(0)
            translate = translate.repeat((x.shape[0], 1))
            x = kornia.translate(x, translate)
        return x


def drop_pixel_channel(x, p=1., drop_p=0.5):
    x = x.clone()
    cwh_list = []
    for cwh in x:
        wh_list = []
        for wh in cwh:
            if np.random.rand() < p:
                mask = torch.rand_like(wh) < drop_p
                mask[:,0] = False
                mask[:,-1] = False
                mask[0,:] = False
                mask[-1,:] = False

                right_mask = torch.zeros_like(mask).bool()
                right_mask[:,1:] = mask[:,:-1]

                # wh[mask] = wh.median()
                left_mask = torch.zeros_like(mask).bool()
                left_mask[:,:-1] = mask[:,1:]

                up_mask = torch.zeros_like(mask).bool()
                up_mask[1:,:] = mask[:-1,:]

                down_mask = torch.zeros_like(mask).bool()
                down_mask[:-1, :] = mask[1:, :]

                wh[mask] = 0.25*(wh[left_mask]+wh[right_mask]+wh[up_mask]+wh[down_mask])
                # wh = blur(wh.unsqueeze(0).unsqueeze(0))
                # wh = wh.squeeze(0).squeeze(0)
            wh_list.append(wh)

        wh_list = torch.stack(wh_list)
        cwh_list.append(wh_list)
    cwh_list = torch.stack(cwh_list)

    return cwh_list
class DropPixel_channel():
    def __init__(self, p=1., drop_rate=0.1):
        super(DropPixel_channel, self).__init__()
        self.p = p
        self.drop_rate = drop_rate

    def __call__(self, x):

        x = drop_pixel_channel(x, self.p, self.drop_rate)
        # x = drop_pixel_channel(x, self.p, self.drop_rate)
        return x


class DWT_Denoise():
    '''
        set the LH,HL,HH sub map to  0
    '''
    def __init__(self, p=1):
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            dwt = DWT()
            idwt = IDWT()
            LL, HH = dwt(x.cpu())
            # HH[0][:,:,-1,:,:] = 0
            # denoise_x = idwt((LL, HH))
            HH = torch.zeros_like(HH[0])
            denoise_x = idwt((LL, [HH]))
            return denoise_x.to(dtype=x.dtype, device=x.device)
        return x


# non-differentiable

class BitDepthReduce():

    def __init__(self, depth=3, p=1):
        '''
        :param depth: keeped bit depth
        '''
        self.depth = depth
        self.p = p

    def __call__(self, x):

        if np.random.rand() < self.p:
            x = torch.round(x*255)
            x = x.to(dtype=torch.uint8)

            shift = 8 - self.depth
            x = (x>>shift) << shift

            x = x.to(dtype=torch.float32) / 255

        return x

class JPEG():
    def __init__(self, quality=75, p=1):
        self.quality = quality
        self.p = p

    def _compression(self, x):
        x = np.transpose(x, (0, 2, 3, 1))
        res = []
        for arr in x:
            pil_image = PIL.Image.fromarray((arr * 255.0).astype(np.uint8))
            f = BytesIO()
            pil_image.save(f, format='jpeg', quality=self.quality)  # quality level specified in paper
            jpeg_image = np.asarray(PIL.Image.open(f)).astype(np.float32) / 255.0
            res.append(jpeg_image)

        x = np.stack(res, axis=0)
        return np.transpose(x, (0, 3, 1, 2))

    def __call__(self, x):

        if np.random.rand() < self.p:
            x_clone = x.detach().cpu().numpy()
            x_clone = self._compression(x_clone)

            return torch.from_numpy(x_clone).to(dtype=x.dtype, device=x.device)

        return x


# CV2, the cv2 default input is BGR, the value is between [0,255]
class NonLocalMean():
    def __init__(self,  filter_strength, patch_size, searchWindowSize, p=1):
        '''
        :param filter_strength:  filter strength. Higher h value removes noise better, but removes details of image also. (10 is ok)
        :param patch_size:should be odd. (recommended 7)
        :param searchWindowSize: should be odd. (recommended 21)
        :param p:
        '''
        self.filter_strength = filter_strength
        self.patch_size = patch_size
        self.searchWindowSize = searchWindowSize
        self.p = p

    def __call__(self, x):

        if np.random.rand() < self.p:

            datas = kornia.rgb_to_bgr(x.clone())
            datas = kornia.tensor_to_image(datas)

            datas = np.round(datas*255)
            datas = datas.astype(np.uint8)

            res_list = []
            for img in datas:
                denoise_img = cv2.fastNlMeansDenoisingColored(img, None, self.filter_strength,
                                                              self.filter_strength, self.patch_size, self.searchWindowSize)
                res_list.append(denoise_img)

            res_list = np.array(res_list)

            datas = kornia.image_to_tensor(res_list)
            datas = kornia.bgr_to_rgb(datas)
            datas = datas.float() / 255

            return datas.to(dtype=x.dtype, device=x.device)

        return x

# kernel size must be odd
class MedianBlur_CV():
    def __init__(self, ksize, p=1):

        self.ksize = ksize
        self.p = p

    def __call__(self, x):

        if np.random.rand() < self.p:

            datas = kornia.rgb_to_bgr(x.clone())
            datas = kornia.tensor_to_image(datas)

            datas = np.round(datas * 255)
            datas = datas.astype(np.uint8)

            res_list = []
            for img in datas:
                denoise_img = cv2.medianBlur(img, ksize=self.ksize)
                res_list.append(denoise_img)

            res_list = np.array(res_list)

            datas = kornia.image_to_tensor(res_list)
            datas = kornia.bgr_to_rgb(datas)
            datas = datas.float() / 255

            return datas.to(dtype=x.dtype, device=x.device)

        return x

# imgaug transforms

# B, W, H， C and dtype uint8, the value is between [0,255].
class MedianBlur_AUG():
    def __init__(self, ksize, p=1):
        self.p = p
        self.aug = iaa.MedianBlur(k=ksize)

    def __call__(self, x):
        if np.random.rand() < self.p:

            datas = kornia.tensor_to_image(x)
            datas = np.round(datas * 255)
            datas = datas.astype(np.uint8)

            datas = self.aug(images=datas)
            datas = kornia.image_to_tensor(datas)
            datas = datas.float() / 255

            return datas.to(dtype=x.dtype, device=x.device)

        return x
#
# class ScaleXY_AUG():
#     def __init__(self, x=(1.0, 1.5), y=(1.0, 1.5), p=1):
#         self.p = p
#
#
#     def __call__(self, x):
#         if np.random.rand() < self.p:
#
#             x = np.random.uniform(x[0], x[1], size=())
#             self.aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})
#
#             datas = kornia.tensor_to_image(x)
#             datas = np.round(datas * 255)
#             datas = datas.astype(np.uint8)
#
#             datas = self.aug(images=datas)
#             datas = kornia.image_to_tensor(datas)
#             datas = datas.float() / 255
#
#             return datas.to(dtype=x.dtype, device=x.device)
#
#         return x

class GaussianNoise_AUG():
    def __init__(self, severity, p=1):
        self.p = p
        self.aug = iaa.imgcorruptlike.GaussianNoise(severity=severity)

    def __call__(self, x):
        if np.random.rand() < self.p:

            datas = kornia.tensor_to_image(x)
            datas = np.round(datas * 255)
            datas = datas.astype(np.uint8)

            datas = self.aug(images=datas)
            datas = kornia.image_to_tensor(datas)
            datas = datas.float() / 255

            return datas.to(dtype=x.dtype, device=x.device)

        return x

class CutOut_AUG():
    def __init__(self, numbers, size=0.1, fill_mode="constant", p=1):
        '''

        :param numbers:  numbers: tuple (a,b) or int, if type is tuple,
                            it random select a number from (a, b)
        :param size: ratio, the area of cut / total area
        :param fill_mode: constant or gaussian
        :param p:
        '''

        self.p = p
        self.aug = iaa.Cutout(nb_iterations=numbers, size=size,fill_mode=fill_mode,
                              squared=False, fill_per_channel=True)

    def __call__(self, x):
        if np.random.rand() < self.p:

            datas = kornia.tensor_to_image(x)
            datas = np.round(datas * 255)
            datas = datas.astype(np.uint8)

            datas = self.aug(images=datas)
            datas = kornia.image_to_tensor(datas)
            datas = datas.float() / 255

            return datas.to(dtype=x.dtype, device=x.device)

        return x


class CoarseDropout_AUG():
    def __init__(self, pixel_percent, size_percent, p=1):
        '''
        :param pixel_percent: tuple or int drop pixel / total pixel
        :param size_percent: tuple or int, the smaller this value, the bigger the drop area
        :param p:
        '''
        self.p = p
        self.aug = iaa.CoarseDropout(pixel_percent, size_percent=size_percent)

    def __call__(self, x):
        if np.random.rand() < self.p:
            datas = kornia.tensor_to_image(x)
            datas = np.round(datas * 255)
            datas = datas.astype(np.uint8)

            datas = self.aug(images=datas)
            datas = kornia.image_to_tensor(datas)
            datas = datas.float() / 255

            return datas.to(dtype=x.dtype, device=x.device)

        return x



class CropResizePad_AUG():
    def __init__(self, p=1):
        self.p = p
    def __call__(self, x):
        if np.random.rand() < self.p:

            h = np.random.randint(224, 240)
            w = np.random.randint(224, 240)
            rand_crop = iaa.CropToFixedSize(width=w, height=h)

            h = np.random.randint(267, 331)
            w = np.random.randint(267, 331)
            rand_resize = iaa.Resize({"height": h, "width": w})

            pad = iaa.PadToFixedSize(width=331, height=331)

            resize = iaa.Resize({"height": 299, "width": 299})

            datas = kornia.tensor_to_image(x)
            datas = np.round(datas * 255)
            datas = datas.astype(np.uint8)

            datas = rand_crop(images=datas)
            datas = rand_resize(images=datas)
            datas = pad(images=datas)
            datas = resize(images=datas)

            datas = kornia.image_to_tensor(np.array(datas))
            datas = datas.float() / 255

            return datas.to(dtype=x.dtype, device=x.device)

        return x



class InputTransformModel(nn.Module):
    def __init__(self, model,  normalize=None, input_trans=None):
        super(InputTransformModel, self).__init__()

        self.model = model
        self.input_trans = input_trans
        self.normalize = normalize

    def __normalize(self, x):
        mean = torch.tensor(self.normalize[0], dtype=x.dtype, device=x.device)
        std = torch.tensor(self.normalize[1], dtype=x.dtype, device=x.device)
        x = (x - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)
        return x

    def forward(self, x):

        if self.input_trans:
            x = self.input_trans(x)

        if self.normalize:
            x = self.__normalize(x)

        return self.model(x)

    def penultimate_forward(self, x):

        if self.input_trans:
            x = self.input_trans(x)

        if self.normalize:
            x = self.__normalize(x)

        return self.model.penultimate_forward(x)


class InputTransformWraper(nn.Module):
    def __init__(self, model,  normalize=None):
        super(InputTransformWraper, self).__init__()

        self.model = model
        self.normalize  = normalize

    def __normalize(self, x):
        mean = torch.tensor(self.normalize[0], dtype=x.dtype, device=x.device)
        std = torch.tensor(self.normalize[1], dtype=x.dtype, device=x.device)
        x = (x - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)
        return x

    def forward(self, x, input_trans=None):

        if input_trans:
            x = input_trans(x)

        if self.normalize:
            x = self.__normalize(x)

        return self.model(x)