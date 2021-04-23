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

import cv2
import numpy as np
import random
import math
import time
import threading

from core import utils

from multiprocessing.dummy import Pool

# region Image resize

def resize_frame(image, target_height=224, target_width=224):
    return __resize_frame(image, target_height, target_width)

def resize_keep_aspect_ratio_max_dim(image, max_dim=None):
    return __resize_keep_aspect_ratio_max_dim(image, max_dim)

def resize_keep_aspect_ratio_min_dim(image, min_dim=None):
    return __resize_keep_aspect_ratio_min_dim(image, min_dim)

def resize_crop(image, target_height=224, target_width=224):
    return __resize_crop(image, target_height, target_width)

def resize_crop_scaled(image, target_height=224, target_width=224):
    return __resize_crop_scaled(image, target_height, target_width)

def resize_keep_aspect_ratio_padded(image, target_height=224, target_width=224):
    return __resize_keep_aspect_ratio_padded(image, target_height, target_width)

def __resize_frame(image, target_height=224, target_width=224):
    """
    Resize to the given dimensions. Don't care about maintaining the aspect ratio of the given image.
    """
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    resized_image = cv2.resize(image, dsize=(target_height, target_width), interpolation=cv2.INTER_LINEAR)
    return resized_image

def __resize_keep_aspect_ratio_max_dim(image, max_dim=224):
    """
    Resize the given image while maintaining the aspect ratio.
    """
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height = image.shape[0]
    width = image.shape[1]

    if height > width:
        target_height = max_dim
        target_width = int(target_height * width / float(height))
    else:
        target_width = max_dim
        target_height = int(target_width * height / float(width))

    resized_image = cv2.resize(image, dsize=(target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def __resize_keep_aspect_ratio_min_dim(image, min_dim=224):
    """
    Resize the given image while maintaining the aspect ratio.
    """
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height = image.shape[0]
    width = image.shape[1]

    if height > width:
        target_width = min_dim
        target_height = int(target_width * height / float(width))
    else:
        target_height = min_dim
        target_width = int(target_height * width / float(height))

    resized_image = cv2.resize(image, dsize=(target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def __resize_crop(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height, target_width), interpolation=cv2.INTER_LINEAR)

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height) / height), target_width), interpolation=cv2.INTER_LINEAR)
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)), interpolation=cv2.INTER_LINEAR)
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length, :]

    resized_image = cv2.resize(resized_image, (target_height, target_width), interpolation=cv2.INTER_LINEAR)
    return resized_image

def __resize_crop_scaled(image, target_height=224, target_width=224):
    # re-scale the image by ratio 3/4 so a landscape or portrait image becomes square
    # then resize_crop it

    # for example, if input image is (height*width) is 400*1000 it will be (400 * 1000 * 3/4) = 400 * 750

    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, _ = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height, target_width), interpolation=cv2.INTER_LINEAR)
    else:

        # first, rescale it, only if the rescale won't bring the scaled dimention to lower than target_dim (= 224)
        scale_factor = 3 / 4.0
        if height < width:
            new_width = int(width * scale_factor)
            if new_width >= target_width:
                image = cv2.resize(image, (new_width, height), interpolation=cv2.INTER_LINEAR)
        else:
            new_height = int(height * scale_factor)
            if new_height >= target_height:
                image = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_LINEAR)

        # now, resize and crop
        height, width, _ = image.shape
        if height < width:
            resized_image = cv2.resize(image, (int(width * float(target_height) / height), target_width), interpolation=cv2.INTER_LINEAR)
            cropping_length = int((resized_image.shape[1] - target_height) / 2)
            resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]

        else:
            resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)), interpolation=cv2.INTER_LINEAR)
            cropping_length = int((resized_image.shape[0] - target_width) / 2)
            resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length, :]

        # this line is important, because sometimes the cropping there is a 1 pixel more
        height, width, _ = resized_image.shape
        if height > target_height or width > target_width:
            resized_image = cv2.resize(resized_image, (target_height, target_width), interpolation=cv2.INTER_LINEAR)

    return resized_image

def __resize_keep_aspect_ratio_padded(image, target_height=224, target_width=224):
    """
    Resize the frame while keeping aspect ratio. Also, to result in an image with the given dimensions, the resized image is zero-padded.
    """

    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    original_height, original_width, _ = image.shape
    original_aspect_ratio = original_height / float(original_width)
    target_aspect_ratio = target_height / float(target_width)

    if target_aspect_ratio >= original_aspect_ratio:
        if original_width >= original_height:
            max_dim = target_width
        else:
            max_dim = int(original_height * target_width / float(original_width))
    else:
        if original_height >= original_width:
            max_dim = target_height
        else:
            max_dim = int(original_width * target_height / float(original_height))

    image = __resize_keep_aspect_ratio_max_dim(image, max_dim=max_dim)

    new_height, new_width, _ = image.shape
    new_aspect_ratio = new_height / float(new_width)

    # do zero-padding for the image (vertical or horizontal)
    img_padded = np.zeros((target_height, target_width, 3), dtype=image.dtype)

    if target_aspect_ratio < new_aspect_ratio:
        # horizontal padding
        y1 = 0
        y2 = new_height
        x1 = int((target_width - new_width) / 2.0)
        x2 = x1 + new_width
    else:
        # vertical padding
        x1 = 0
        x2 = new_width
        y1 = int((target_height - new_height) / 2.0)
        y2 = y1 + new_height

    img_padded[y1:y2, x1:x2, :] = image
    return img_padded

# endregion

# region Image Mosaic

def mosaic_images_horizontally(imgs, offset):
    n_imgs = len(imgs)
    H, W, C = imgs[0].shape
    H_ = H
    W_ = ((W + offset) * n_imgs) - offset
    img_big = np.ones((H_, W_, C), dtype=imgs.dtype) * 255

    for idx_img in range(n_imgs):
        start_idx = idx_img * W if idx_img == 0 else idx_img * (W + offset)
        stop_idx = start_idx + W

        img = imgs[idx_img]
        img_big[:, start_idx:stop_idx, :] = img

    return img_big

def mosaic_images_horizontally_bordered(imgs, border):
    n_imgs = len(imgs)
    H, W, C = imgs[0].shape
    H_ = H + (border * 2)
    W_ = ((W + border) * n_imgs) + 2 * border
    img_big = np.zeros((H_, W_, C), dtype=imgs.dtype) * 255

    for idx_img in range(n_imgs):
        start_idx = idx_img * (W + border) + border
        stop_idx = start_idx + W

        img = imgs[idx_img]
        img_big[border: H + border, start_idx:stop_idx, :] = img

    return img_big

def mosaic_images_vertically(imgs, offset):
    n_imgs = len(imgs)
    H, W, C = imgs[0].shape
    W_ = W
    H_ = ((H + offset) * n_imgs) - offset
    img_big = np.ones((H_, W_, C), dtype=imgs.dtype) * 255

    for idx_img in range(n_imgs):
        start_idx = idx_img * H if idx_img == 0 else idx_img * (H + offset)
        stop_idx = start_idx + H

        img = imgs[idx_img]
        img_big[start_idx:stop_idx, :, :] = img

    return img_big

def paper_mosaic(imgs_root_path, img_out_path):
    imgs = np.array([cv2.imread(n) for n in utils.file_pathes(imgs_root_path, is_natsort=True)])
    img = mosaic_images_horizontally(imgs, offset=-20)
    H, W, C = img.shape
    H = int(H * W / 1000.0)
    img = cv2.resize(img, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(img_out_path, img)

# endregion

# region Test Cases

def test_random_crop():
    img_path = '/home/nour/Pictures/cees-snoek.jpg'
    img = cv2.imread(img_path)
    img = resize_keep_aspect_ratio_min_dim(img, 256)

    h, w, _ = img.shape
    col = (0, 0, 255)

    for i in range(1000):
        x1 = random.randint(0, w - 224)
        y1 = random.randint(0, h - 224)

        x2 = x1 + 224
        y2 = y1 + 224

        img_ = img.copy()
        # img_ = img_[y1:y2, x1:x2]
        cv2.rectangle(img_, (x1, y1), (x2, y2), col, 5)

        cv2.imshow('Window_Name', img_)
        cv2.waitKey(80)

def test_five_crops():
    img_path = '/home/nour/Pictures/cees-snoek.jpg'
    img = cv2.imread(img_path)
    img = resize_keep_aspect_ratio_min_dim(img, 256)

    h, w, _ = img.shape

    # crops are top-left, top-right, bottom-left, bottom-right, center
    crop_x1 = w - 224
    crop_y1 = h - 224
    crop5_x1 = int((w - 224) / 2.0)
    crop5_y1 = int((h - 224) / 2.0)
    print(w, h)
    print(crop5_x1, crop5_y1)

    # crop 1
    x1, y1, x2, y2 = 0, 0, 224, 224
    img_ = img[y1:y2, x1:x2]
    cv2.imshow('Window_Name1', img_)

    # crop 2
    x1, y1, x2, y2 = crop_x1, 0, w, 224
    img_ = img[y1:y2, x1:x2]
    cv2.imshow('Window_Name2', img_)

    # crop 3
    x1, y1, x2, y2 = 0, crop_y1, 224, h
    img_ = img[y1:y2, x1:x2]
    cv2.imshow('Window_Name3', img_)

    # crop 4
    x1, y1, x2, y2 = crop_x1, crop_y1, w, h
    img_ = img[y1:y2, x1:x2]
    cv2.imshow('Window_Name4', img_)

    # crop 5
    x1, y1, x2, y2 = crop5_x1, crop5_y1, crop5_x1 + 224, crop5_y1 + 224
    img_ = img[y1:y2, x1:x2]
    cv2.imshow('Window_Name5', img_)

    print(x1, y1, x2, y2)

    cv2.waitKey(100000)

def test_async_image_reader_hmdb():
    mean = vgg.get_mean_pixel(is_imagenet_vgg=True)

    img_root_path = '/home/nour/Documents/Datasets/HMDB/frames/smile/A_Beautiful_Mind_2_smile_h_nm_np1_fr_goo_2'
    img_root_path = '/home/nour/Documents/Datasets/HMDB/frames/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0'
    img_root_path = '/home/nour/Documents/Datasets/HMDB/frames/catch/Faith_Rewarded_catch_f_cm_np1_fr_med_10'
    img_root_path = '/home/nour/Documents/Datasets/HMDB/frames/sit/50_FIRST_DATES_sit_f_cm_np1_fr_med_24'
    img_pathes = utils.file_pathes(img_root_path)

    img_pathes = ['/home/nour/Pictures/cees-snoek.jpg'] * 200

    asyncReader = AsyncImageReaderHmdb(100, img_pathes, vgg_mean=mean, is_training=False)
    asyncReader.load_batch(1)

    while asyncReader.is_busy():
        time.sleep(0.1)

    mean = vgg.get_mean_pixel(is_imagenet_vgg=True)
    imgs = asyncReader.get_batch()
    for img in imgs:
        img[:, :, 0] += mean[0]
        img[:, :, 1] += mean[1]
        img[:, :, 2] += mean[2]
        img = img.astype(np.uint8)
        cv2.imshow('Images', img)
        cv2.waitKey(500)

def test_async_image_reader_places():
    mean = vgg.get_mean_pixel(is_imagenet_vgg=True)

    img_root_path = '/home/nour/Documents/Datasets/HMDB/frames/smile/A_Beautiful_Mind_2_smile_h_nm_np1_fr_goo_2'
    img_root_path = '/home/nour/Documents/Datasets/HMDB/frames/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0'
    img_root_path = '/home/nour/Documents/Datasets/HMDB/frames/catch/Faith_Rewarded_catch_f_cm_np1_fr_med_10'
    img_root_path = '/home/nour/Documents/Datasets/HMDB/frames/sit/50_FIRST_DATES_sit_f_cm_np1_fr_med_24'

    result_root_path = '/home/nour/Desktop/test_result/'

    img_pathes = utils.file_pathes(img_root_path)
    img_pathes = np.array(img_pathes)
    img_pathes = img_pathes[0:10]

    img_pathes = np.array(['/home/nour/Pictures/cees-snoek.jpg', '/home/nour/Pictures/540115_354489667925978_1963943715_n.jpg', '/home/nour/Pictures/QUVA_IMG_1.jpg'])

    batch_size = 7
    n_imgs = len(img_pathes)
    n_batches = int(n_imgs / float(batch_size))
    n_batches = n_batches if n_imgs % batch_size == 0 else n_batches + 1

    async_reader = AsyncImageReaderPlaces(vgg_mean=mean, is_training=False, is_multi_test_crops=False)
    count = 0
    for idx_batch in range(n_batches):
        start_idx = idx_batch * batch_size
        stop_idx = (idx_batch + 1) * batch_size
        img_pathes_batch = img_pathes[start_idx:stop_idx]
        async_reader.load_batch(img_pathes_batch)

        t1 = time.time()
        while async_reader.is_busy():
            time.sleep(0.1)
        t2 = time.time()

        print(len(img_pathes_batch), t2 - t1)
        mean = vgg.get_mean_pixel(is_imagenet_vgg=True)
        imgs = async_reader.get_batch()
        for img in imgs:
            img[:, :, 0] += mean[0]
            img[:, :, 1] += mean[1]
            img[:, :, 2] += mean[2]
            img = img.astype(np.uint8)
            count += 1
            img_path = '/home/nour/Desktop/test_result/%04d.jpg' % (count)
            cv2.imwrite(img_path, img)
            # cv2.imshow('Images', img)
            # cv2.waitKey(100)

def test_async_image_reader_simple():
    mean = vgg.get_mean_pixel(is_imagenet_vgg=True)

    img_root_path = '/home/nour/Documents/Datasets/HMDB/frames/smile/A_Beautiful_Mind_2_smile_h_nm_np1_fr_goo_2'
    img_root_path = '/home/nour/Documents/Datasets/HMDB/frames/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0'
    img_root_path = '/home/nour/Documents/Datasets/HMDB/frames/catch/Faith_Rewarded_catch_f_cm_np1_fr_med_10'
    img_root_path = '/home/nour/Documents/Datasets/HMDB/frames/sit/50_FIRST_DATES_sit_f_cm_np1_fr_med_24'

    result_root_path = '/home/nour/Desktop/test_result/'

    img_pathes = utils.file_pathes(img_root_path)
    img_pathes = np.array(img_pathes)
    img_pathes = img_pathes[0:10]

    img_pathes = np.array(['/home/nour/Pictures/cees-snoek.jpg', '/home/nour/Pictures/540115_354489667925978_1963943715_n.jpg', '/home/nour/Pictures/QUVA_IMG_1.jpg'])

    batch_size = 7
    n_imgs = len(img_pathes)
    n_batches = int(n_imgs / float(batch_size))
    n_batches = n_batches if n_imgs % batch_size == 0 else n_batches + 1

    async_reader = AsyncImageReaderSimple(vgg_mean=mean, is_training=False)
    count = 0
    for idx_batch in range(n_batches):
        start_idx = idx_batch * batch_size
        stop_idx = (idx_batch + 1) * batch_size
        img_pathes_batch = img_pathes[start_idx:stop_idx]
        async_reader.load_batch(img_pathes_batch)

        t1 = time.time()
        while async_reader.is_busy():
            time.sleep(0.1)
        t2 = time.time()

        print(len(img_pathes_batch), t2 - t1)
        mean = vgg.get_mean_pixel(is_imagenet_vgg=True)
        imgs = async_reader.get_batch()
        for img in imgs:
            img[:, :, 0] += mean[0]
            img[:, :, 1] += mean[1]
            img[:, :, 2] += mean[2]
            img = img.astype(np.uint8)
            count += 1
            img_path = '/home/nour/Desktop/test_result/%04d.jpg' % (count)
            cv2.imwrite(img_path, img)
            # cv2.imshow('Images', img)
            # cv2.waitKey(100)

def __read_images(pathes):
    min_size = 256
    max_size = 320
    max_ratio = 1.6
    h_crop = 224
    w_crop = 224

    now_hight = 256
    now_width = int(now_hight * 340.0 / 256.0)

    assert (now_hight >= h_crop);
    assert (now_width >= w_crop);

    imgs = np.zeros((len(pathes), 224, 224, 3), dtype=np.float)
    for idx, p in enumerate(pathes):
        img = cv2.imread(p)
        h, w, _ = img.shape

        # get ratio
        side_length = min_size if (min_size == max_size) else np.random.randint(low=min_size, high=max_size)
        ratio = side_length / float(w) if h > w else side_length / float(h)
        ratio = min(max_ratio, ratio)

        # get new resize dims
        h_new = int(h * ratio)
        w_new = int(w * ratio)

        # resize
        img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        # random crop
        h_off = np.random.randint(low=0, high=h_new - h_crop)
        w_off = np.random.randint(low=0, high=w_new - w_crop)
        y1 = h_off
        y2 = h_off + h_crop
        x1 = w_off
        x2 = w_off + w_crop

        img = img[y1:y2, x1:x2]
        h, w, _ = img.shape
        assert h == h_crop
        assert w == w_crop

        # covert to float
        img = img.astype(np.float32)

        # normalize such that values range from -1 to 1
        img /= float(127.5)
        img -= 1.0
        # convert from bgr to rgb
        img = img[:, :, (2, 1, 0)]
        print(img.shape)

        imgs[idx] = img

    return imgs

# endregion

# region Compare Libraries for Reading Images

#
# import cv2
# import time
# from core import utils
# from PIL import Image
# # import gdal
# from osgeo import gdal
#
# img_path = '/home/nour/Pictures/arnold.jpg'
# img_path = '/home/nhussein/arnold.jpg'
# data_path = '/home/nour/Pictures/data.pkl'
# img_str_path = '/home/nour/Pictures/data_bytes.pkl'
#
#
# duration = 0.0
# n = 100
# for i in range(n):
#     t1 = time.time()
#     img = gdal.Open(img_path, gdal.GA_ReadOnly)
#     img = img.GetRasterBand(1).ReadAsArray()
#     # img = cv2.imread(img_path)
#     # img = utils.pkl_load(data_path)
#     # img = Image.open(img_path)
#     # img = np.asarray(Image.open(img_path))
#     t2 = time.time()
#     d = t2 - t1
#     duration += d
# print duration

# endregion
