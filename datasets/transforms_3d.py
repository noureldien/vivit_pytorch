# !/usr/bin/env python
# -*- coding: UTF-8 -*-

########################################################################
# GNU General Public License v3.0
# GNU GPLv3
# Copyright (c) 2019, Noureldien Hussein
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
########################################################################

import random
import numbers
import math
import cv2
import collections
import numpy as np
from PIL import ImageOps, Image, ImageFilter
from joblib import Parallel, delayed

import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms

# region Spatial Transforms

class Padding:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, img):
        return ImageOps.expand(img, border=self.pad, fill=0)

class Resize:
    def __init__(self, size, interpolation=Image.NEAREST):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        img1 = imgmap[0]
        if isinstance(self.size, int):
            w, h = img1.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return imgmap
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
        else:
            return [i.resize(self.size, self.interpolation) for i in imgmap]

class CenterCrop:
    def __init__(self, size, ):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]

class Rotation:
    """
    Rotate the frames in the video segment by the given angle.
    """

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, imgmap):
        imgmap = [i.rotate(self.degree, expand=False) for i in imgmap]

        return imgmap

# endregion

# region Spatial Transforms (Random)

class RandomCrop:
    def __init__(self, size, p=0.8, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            if self.consistent:
                if random.random() < self.threshold:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                else:
                    x1 = int(round((w - tw) / 2.))
                    y1 = int(round((h - th) / 2.))
                return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]
            else:
                result = []
                for i in imgmap:
                    if random.random() < self.threshold:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                    else:
                        x1 = int(round((w - tw) / 2.))
                        y1 = int(round((h - th) / 2.))
                    result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                return result
        else:
            return imgmap

class RandomResizedCrop:
    def __init__(self, resize_dim, crop_dim, interpolation=Image.BILINEAR, consistent=True, p=1.0):
        self.resize_dim = resize_dim
        self.crop_dim = crop_dim
        self.interpolation = interpolation
        self.consistent = consistent
        self.threshold = p

        self.resize = Resize(self.resize_dim, interpolation=self.interpolation)
        self.center_crop = CenterCrop(self.crop_dim)

    def __call__(self, imgmap):
        img1 = imgmap[0]
        if random.random() < self.threshold:  # do RandomSizedCrop
            for attempt in range(10):
                area = img1.size[0] * img1.size[1]
                target_area = random.uniform(0.5, 1) * area
                aspect_ratio = random.uniform(3. / 4, 4. / 3)

                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))

                if self.consistent:
                    if random.random() < 0.5:
                        w, h = h, w
                    if w <= img1.size[0] and h <= img1.size[1]:
                        x1 = random.randint(0, img1.size[0] - w)
                        y1 = random.randint(0, img1.size[1] - h)
                        imgmap = [i.crop((x1, y1, x1 + w, y1 + h)) for i in imgmap]
                        for i in imgmap:
                            assert (i.size == (w, h))
                        imgmap = [i.resize((self.crop_dim, self.crop_dim), self.interpolation) for i in imgmap]
                        return imgmap
                else:
                    result = []
                    for i in imgmap:
                        if random.random() < 0.5:
                            w, h = h, w
                        if w <= img1.size[0] and h <= img1.size[1]:
                            x1 = random.randint(0, img1.size[0] - w)
                            y1 = random.randint(0, img1.size[1] - h)
                            result.append(i.crop((x1, y1, x1 + w, y1 + h)))
                            assert (result[-1].size == (w, h))
                        else:
                            result.append(i)

                    assert len(result) == len(imgmap)
                    imgmap = [i.resize((self.crop_dim, self.crop_dim), self.interpolation) for i in result]
                    return imgmap

            # Fallback
            imgmap = self.center_crop(self.resize(imgmap))
            return imgmap
        else:
            # don't do RandomSizedCrop, do CenterCrop
            imgmap = self.center_crop(imgmap)
            return imgmap

class RandomHorizontalFlip:
    def __init__(self, p=0.5, consistent=True):
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        if self.consistent:
            if random.random() < self.threshold:
                return [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for i in imgmap:
                if random.random() < self.threshold:
                    result.append(i.transpose(Image.FLIP_LEFT_RIGHT))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result

class RandomVerticalFlip:
    def __init__(self, p=0.5, consistent=True):
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        if self.consistent:
            if random.random() < self.threshold:
                return [i.transpose(Image.FLIP_TOP_BOTTOM) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for i in imgmap:
                if random.random() < self.threshold:
                    result.append(i.transpose(Image.FLIP_TOP_BOTTOM))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result

class RandomRotation:
    """
    Rotate the frames in the video segment by a randomly sampled angle from a certain range.
    """

    def __init__(self, degree, p=1.0, consistent=True):
        self.degree = degree
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        if random.random() < self.threshold:
            if self.consistent:
                deg = self.__rand_int()
                imgmap = [i.rotate(deg, expand=False) for i in imgmap]
            else:
                imgmap = [i.rotate(self.__rand_int(), expand=False) for i in imgmap]

        return imgmap

    def __rand_int(self):
        num = np.random.randint(-self.degree, self.degree, 1)[0]
        return num

class RandomTranslatedCrop:
    """
    Translate the center crop of the image by certain displacement ratios (dx, dy).
    These two ratios are independently and randomly sampled from a certain range.
    """

    def __init__(self, translation_ratio, size, p=1.0, consistent=True):
        self.translation_ratio = translation_ratio
        self.consistent = consistent
        self.threshold = p

        # target size
        if isinstance(size, numbers.Number):
            self.th = self.tw = int(size)
        else:
            self.th, self.tw = size

        self.th_half = int(self.th / 2.0)
        self.tw_half = int(self.tw / 2.0)

    def __call__(self, imgmap):
        if random.random() < self.threshold:
            if self.consistent:
                dx = self.__rand_float()
                dy = self.__rand_float()
                imgmap = [self.__translated_center_crop(i, dx, dy) for i in imgmap]
            else:
                imgmap = [self.__translated_center_crop(i, self.__rand_float(), self.__rand_float()) for i in imgmap]

        return imgmap

    def __rand_float(self):
        number = np.random.uniform(-self.translation_ratio, self.translation_ratio, 1)[0]
        return number

    def __translated_center_crop(self, img, dx, dy):
        """
        Translate the center crop of the image by by x, y pixels,
        where x = dx * img_width, y = dy * img_height
        """

        w, h = img.size
        tw = self.tw
        th = self.th

        tw_half = int(tw / 2.0)
        th_half = int(th / 2.0)

        # translate x0 and y0
        x0 = int((w / 2.0) + (w * dx))
        y0 = int((h / 2.0) + (h * dy))

        x1 = x0 - tw_half
        y1 = y0 - th_half
        x2 = x1 + tw
        y2 = y1 + th

        if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
            img_target = img.crop((x1, y1, x2, y2))
        else:

            # x1_s, x2_s, .... => coordinates of the crop in the source image
            # x1_t, x2_t, .... => coordinates of the crop target the source image
            if x1 < 0:
                x1_s = 0
                x1_t = tw - x2
            else:
                x1_s = x1
                x1_t = 0

            if y1 < 0:
                y1_s = 0
                y1_t = th - y2
            else:
                y1_s = y1
                y1_t = 0

            if x2 > w:
                x2_s = w
                x2_t = w - x1
            else:
                x2_s = x2
                x2_t = tw

            if y2 > h:
                y2_s = h
                y2_t = h - y1
            else:
                y2_s = y2
                y2_t = th

            img_target = np.zeros((th, tw, 3), dtype=np.uint8)
            img = np.asarray(img)
            img_target[y1_t:y2_t, x1_t:x2_t] = img[y1_s:y2_s, x1_s:x2_s]
            img_target = Image.fromarray(img_target)

        return img_target

class RandomResizedCenterCrop:
    """
    Scale the center crop of the frame in the video segment by a randomly sampled scaling factor from a certain range.
    Also, change the aspect ratio of the center crop by a randomly sampled aspect ratio factor from another range.
    Then, resized the center crop to the acceptable size.
    """

    def __init__(self, resize_dim, crop_dim, scale, aspect, p=1.0, consistent=True, interpolation=Image.BILINEAR):
        self.resize_dim = resize_dim
        self.crop_dim = crop_dim
        self.scale = scale
        self.aspect = aspect
        self.consistent = consistent
        self.threshold = p
        self.interpolation = interpolation
        self.n_attempts = 10

        self.is_scale = scale is not None
        self.is_aspect = aspect is not None

        self.resize = Resize(self.resize_dim, interpolation=self.interpolation)
        self.center_crop = CenterCrop(self.crop_dim)

    def __call__(self, imgmap):

        img1 = imgmap[0]
        if random.random() < self.threshold:
            for attempt in range(self.n_attempts):
                area = img1.size[0] * img1.size[1]

                if self.consistent:

                    # w, h of the center crop
                    w, h = self.__random_dims(area)

                    if random.random() < 0.5:
                        w, h = h, w
                    if w <= img1.size[0] and h <= img1.size[1]:
                        x1 = int(round((img1.size[0] - w) / 2.0))
                        y1 = int(round((img1.size[1] - h) / 2.0))
                        imgmap = [i.crop((x1, y1, x1 + w, y1 + h)).resize((self.crop_dim, self.crop_dim), self.interpolation) for i in imgmap]
                        return imgmap
                else:
                    result = []
                    for i in imgmap:

                        # w, h of the center crop
                        w, h = self.__random_dims(area)

                        if random.random() < 0.5:
                            w, h = h, w
                        if w <= img1.size[0] and h <= img1.size[1]:
                            x1 = int(round((img1.size[0] - w) / 2.0))
                            y1 = int(round((img1.size[1] - h) / 2.0))
                            result.append(i.crop((x1, y1, x1 + w, y1 + h)))
                        else:
                            result.append(i)

                    assert len(result) == len(imgmap)
                    imgmap = [i.resize((self.crop_dim, self.crop_dim), self.interpolation) for i in result]
                    return imgmap

            # Fallback
            imgmap = self.center_crop(self.resize(imgmap))
            return imgmap
        else:
            # don't do resize, just center crop
            imgmap = self.center_crop(imgmap)
            return imgmap

    def __random_dims(self, area):

        # sample area and aspect ratio
        target_area = random.uniform(self.scale, 1) if self.is_scale else 1
        target_area = target_area * area

        aspect_ratio = random.uniform(-self.aspect, self.aspect) if self.is_aspect else 0
        aspect_ratio = aspect_ratio + 1

        # w, h of the center crop
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        dims = (w, h)
        return dims

class RandomResizedScale:
    """
    Scale the frame in the video segment by a randomly sampled scaling factor from a certain range.
    """

    def __init__(self, resize_dim, scale, p=1.0, consistent=True, interpolation=Image.BILINEAR):
        self.resize_dim = resize_dim
        self.scale = scale
        self.consistent = consistent
        self.threshold = p
        self.interpolation = interpolation

        self.resize = Resize(self.resize_dim, interpolation=self.interpolation)

    def __call__(self, imgmap):

        img1 = imgmap[0]
        if random.random() < self.threshold:
            ow, oh = img1.size[0], img1.size[1]
            if self.consistent:
                w, h = self.__random_dims(ow, oh)
                imgmap = [i.resize((w, h), self.interpolation) for i in imgmap]
                return imgmap
            else:
                result = []
                for i in imgmap:
                    w, h = self.__random_dims(ow, oh)
                    i.resize((w, h), self.interpolation)
                    result.append(i)
                return imgmap

        else:
            imgmap = self.resize(imgmap)
            return imgmap

    def __random_dims(self, w, h):

        # get random scale factor
        scale = 1 + random.uniform(-self.scale, self.scale)

        # scale the dimensions
        if w < h:
            tw = self.resize_dim * scale
            th = h * tw / float(w)
        else:
            th = self.resize_dim * scale
            tw = w * th / float(h)

        dims = (int(round(tw)), int(round(th)))
        return dims

class RandomResizedAspect:
    """
    Change the aspect ratio of the center crop by a randomly sampled aspect ratio factor from another range.
    """

    def __init__(self, resize_dim, aspect, p=1.0, consistent=True, interpolation=Image.BILINEAR):
        self.resize_dim = resize_dim
        self.aspect = aspect
        self.consistent = consistent
        self.threshold = p
        self.interpolation = interpolation

        self.resize = Resize(self.resize_dim, interpolation=self.interpolation)

    def __call__(self, imgmap):

        img1 = imgmap[0]
        if random.random() < self.threshold:
            ow, oh = img1.size[0], img1.size[1]
            if self.consistent:
                w, h = self.__random_dims(ow, oh)
                imgmap = [i.resize((w, h), self.interpolation) for i in imgmap]
                return imgmap
            else:
                result = []
                for i in imgmap:
                    w, h = self.__random_dims(ow, oh)
                    i.resize((w, h), self.interpolation)
                    result.append(i)
                return imgmap
        else:
            imgmap = self.resize(imgmap)
            return imgmap

    def __random_dims(self, ow, oh):

        area = ow * oh
        aspect = 1 + random.uniform(-self.aspect, self.aspect)

        if ow < oh:
            w = math.sqrt(area * aspect)
            h = math.sqrt(area / aspect)
            tw = self.resize_dim
            th = int(round(h * tw / float(w)))
        else:
            h = math.sqrt(area * aspect)
            w = math.sqrt(area / aspect)
            th = self.resize_dim
            tw = int(round(w * th / float(h)))

        dims = (tw, th)
        return dims

# endregion

# region Misc Transforms

class ToTensor:
    def __call__(self, imgmap):
        totensor = transforms.ToTensor()
        return [totensor(i) for i in imgmap]

class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def __call__(self, imgmap):
        imgmap = [self.normalize(i) for i in imgmap]
        return imgmap

class Stack:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        x = torch.stack(x, dim=self.dim)
        return x

# endregion