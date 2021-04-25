"""
Code borrowed from:
https://github.com/TwentyBN/smth-smth-v2-baseline-with-models/blob/master/data_augmentor.py
"""

import json
import numpy as np
from datasets.transforms_3d import *

class Augmentor(object):
    def __init__(self, augmentation_mappings_json=None, augmentation_types_todo=None, fps_jitter_factors=[1, 0.75, 0.5]):
        self.augmentation_mappings_json = augmentation_mappings_json
        self.augmentation_types_todo = augmentation_types_todo
        self.fps_jitter_factors = fps_jitter_factors

        # read json to get the mapping dict
        self.augmentation_mapping = self.read_augmentation_mapping(self.augmentation_mappings_json)
        self.augmentation_transforms = self.define_augmentation_transforms()

    def __call__(self, imgs, label):
        if not self.augmentation_mapping:
            return imgs, label
        else:
            candidate_augmentations = {"same": label}
            for candidate in self.augmentation_types_todo:
                if candidate == "jitter_fps":
                    continue
                if label in self.augmentation_mapping[candidate]:
                    if isinstance(self.augmentation_mapping[candidate], list):
                        candidate_augmentations[candidate] = label
                    elif isinstance(self.augmentation_mapping[candidate], dict):
                        candidate_augmentations[candidate] = self.augmentation_mapping[candidate][label]
                    else:
                        print("Something wrong with data type specified in "
                              "augmentation file. Please check!")
            augmentation_chosen = np.random.choice(list(candidate_augmentations.keys()))
            imgs = self.augmentation_transforms[augmentation_chosen](imgs)
            label = candidate_augmentations[augmentation_chosen]

            return imgs, label

    def read_augmentation_mapping(self, path):
        if path:
            with open(path, "rb") as fp:
                mapping = json.load(fp)
        else:
            mapping = None
        return mapping

    def define_augmentation_transforms(self, ):
        augmentation_transforms = {}
        augmentation_transforms["same"] = IdentityTransform()
        augmentation_transforms["left/right"] = RandomHorizontalFlipVideo(1)
        augmentation_transforms["left/right agnostic"] = RandomHorizontalFlipVideo(1)
        augmentation_transforms["reverse time"] = RandomReverseTimeVideo(1)
        augmentation_transforms["reverse time agnostic"] = RandomReverseTimeVideo(0.5)

        return augmentation_transforms

    def jitter_fps(self, framerate):
        if self.augmentation_types_todo and "jitter_fps" in self.augmentation_types_todo:
            jitter_factor = np.random.choice(self.fps_jitter_factors)
            return int(jitter_factor * framerate)
        else:
            return framerate
