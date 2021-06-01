'''Dataset class to feed to Pytorch dataloader API'''

import os

import numpy as np
import torch
import imageio

import config


class CellDataset(object):
    '''Pytorch compatible dataset class. Fetches HPA
    cell-level images and corresponding weak labels.
    '''
    def __init__(self, images, targets, img_root, augmentations=None):
        self.images = images
        self.targets = targets
        self.img_root = img_root
        self.augmentations = augmentations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images[idx]
        img_channels = self._fetch_channels(img_id)
        img = self._channels_2_array(img_channels)
        if self.augmentations:
            img = self.augmentations(image=img)['image']
        # Switch to channel first indexing for pytorch (speed reasons)
        features = np.transpose(img, (2, 0, 1)).astype(np.float32)
        target = self.targets[idx]  # Grab target vector

        return {
            'image': torch.tensor(features),
            'target': torch.tensor(target)
        }

    def _fetch_channels(self, img_id: str, channel_names=config.CHANNELS):
        'Return absolute path of segmentation channels of a given image id'
        base = os.path.join(self.img_root, img_id)
        return [base + '_' + i  + '.png' for i in channel_names]

    def _channels_2_array(self, img_channels):
        'Return 3D array of pixel values of input image channels'
        r = imageio.imread(img_channels[0])
        g = imageio.imread(img_channels[1])
        b = imageio.imread(img_channels[2])
        pixel_arr = np.dstack((r, g, b))
        return pixel_arr
