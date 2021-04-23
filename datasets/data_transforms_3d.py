"""
Code borrowed from:
https://raw.githubusercontent.com/TwentyBN/smth-smth-v2-baseline-with-models/master/transforms_video.py
"""

import torch
import cv2
import numpy as np
import numbers
import collections
import random

class ComposeMix(object):
    r"""Composes several transforms together. It takes a list of
    transformations, where each element odf transform is a list with 2
    elements. First being the transform function itself, second being a string
    indicating whether it's an "img" or "vid" transform
    Args:
        transforms (List[Transform, "<type>"]): list of transforms to compose.
                                                <type> = "img" | "vid"
    Example:
        >>> transforms.ComposeMix([
        [RandomCropVideo(84), "vid"],
        [torchvision.transforms.ToTensor(), "img"],
        [torchvision.transforms.Normalize(
                   mean=[0.485, 0.456, 0.406],  # default values for imagenet
                   std=[0.229, 0.224, 0.225]), "img"]
    ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        for t in self.transforms:
            if t[1] == "img":
                for idx, img in enumerate(imgs):
                    imgs[idx] = t[0](img)
            elif t[1] == "vid":
                imgs = t[0](imgs)
            else:
                print("Please specify the transform type")
                raise ValueError
        return imgs

class RandomCropVideo(object):
    r"""Crop the given video frames at a random location. Crop location is the
    same for all the frames.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_method (cv2 constant): Method to be used for padding.
    """

    def __init__(self, size, padding=0, pad_method=cv2.BORDER_CONSTANT):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_method = pad_method

    def __call__(self, imgs):
        """
        Args:
            img (numpy.array): Video to be cropped.
        Returns:
            numpy.array: Cropped video.
        """
        th, tw = self.size
        h, w = imgs[0].shape[:2]
        x1 = np.random.randint(0, w - tw)
        y1 = np.random.randint(0, h - th)
        for idx, img in enumerate(imgs):
            if self.padding > 0:
                img = cv2.copyMakeBorder(img, self.padding, self.padding,
                                         self.padding, self.padding,
                                         self.pad_method)
            # sample crop locations if not given
            # it is necessary to keep cropping same in a video
            img_crop = img[y1:y1 + th, x1:x1 + tw]
            imgs[idx] = img_crop
        return imgs

class RandomHorizontalFlipVideo(object):
    """Horizontally flip the given video frames randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            imgs (numpy.array): Video to be flipped.
        Returns:
            numpy.array: Randomly flipped video.
        """
        if random.random() < self.p:
            for idx, img in enumerate(imgs):
                imgs[idx] = cv2.flip(img, 1)
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomReverseTimeVideo(object):
    """Reverse the given video frames in time randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            imgs (numpy.array): Video to be flipped.
        Returns:
            numpy.array: Randomly flipped video.
        """
        if random.random() < self.p:
            imgs = imgs[::-1]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomRotationVideo(object):
    """Rotate the given video frames randomly with a given degree.
    Args:
        degree (float): degrees used to rotate the video
    """

    def __init__(self, degree=10):
        self.degree = degree

    def __call__(self, imgs):
        """
        Args:
            imgs (numpy.array): Video to be rotated.
        Returns:
            numpy.array: Randomly rotated video.
        """
        h, w = imgs[0].shape[:2]
        degree_sampled = np.random.choice(
            np.arange(-self.degree, self.degree, 0.5))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), degree_sampled, 1)

        for idx, img in enumerate(imgs):
            imgs[idx] = cv2.warpAffine(img, M, (w, h))

        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(degree={})'.format(self.degree_sampled)

class IdentityTransform(object):
    """
    Returns same video back
    """

    def __init__(self, ):
        pass

    def __call__(self, imgs):
        return imgs

class Scale(object):
    r"""Rescale the input image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(
            size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (numpy.array): Image to be scaled.
        Returns:
            numpy.array: Rescaled image.
        """
        if isinstance(self.size, int):
            h, w = img.shape[:2]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                if ow < w:
                    return cv2.resize(img, (ow, oh), cv2.INTER_AREA)
                else:
                    return cv2.resize(img, (ow, oh))
            else:
                oh = self.size
                ow = int(self.size * w / h)
                if oh < h:
                    return cv2.resize(img, (ow, oh), cv2.INTER_AREA)
                else:
                    return cv2.resize(img, (ow, oh))
        else:
            return cv2.resize(img, tuple(self.size))

class UnNormalize(object):
    """Unnormalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel x std) + mean
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean).astype('float32')
        self.std = np.array(std).astype('float32')

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if isinstance(tensor, torch.Tensor):
            self.mean = torch.FloatTensor(self.mean)
            self.std = torch.FloatTensor(self.std)

            if (self.std.dim() != tensor.dim() or
                    self.mean.dim() != tensor.dim()):
                for i in range(tensor.dim() - self.std.dim()):
                    self.std = self.std.unsqueeze(-1)
                    self.mean = self.mean.unsqueeze(-1)

            tensor = torch.add(torch.mul(tensor, self.std), self.mean)
        else:
            # Relying on Numpy broadcasting abilities
            tensor = tensor * self.std + self.mean
        return tensor
