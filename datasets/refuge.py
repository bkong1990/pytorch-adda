"""Dataset setting and data loader for USPS.

Modified from
https://github.com/mingyuliutw/CoGAN/blob/master/cogan_pytorch/src/dataset_usps.py
"""

import os
import numpy as np
import glob
import scipy.misc as m

import torch
import torch.utils.data as data

import params


class REFUGE(data.Dataset):
    """USPS Dataset.

    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    def __init__(
        self, 
        train=True, 
        domain='REFUGE_SRC', 
        is_transform=False, 
        augmentations=None,
        img_size=(512, 512),
        img_norm=True
    ):
        """Init USPS dataset."""
        # init params
        self.train = train
        self.is_transform = is_transform
        # Num of Train = 7438, Num ot Test 1860
        self.augmentations = augmentations
        self.dataset_size = None
        self.img_norm = img_norm
        self.domain = domain
        if domain == 'REFUGE_SRC':
            self.img_dir = params.src_image_dir
            self.mask_dir = params.src_mask_dir
        else:
            self.img_dir = params.tgt_image_dir
            self.mask_dir = params.tgt_mask_dir

        self.class_map = {255: 0, 128: 1, 0: 2}
        self.mean = np.array([0, 0, 0])
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )

        self._glob_img_files()
        if self.train:
            self.image_files = self.image_files[: -10]
        else:
            self.image_files = self.image_files[-10:]

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image_file = self.image_files[index]
        label_file = os.path.join(self.mask_dir, os.path.basename(image_file))[:-3] + 'bmp'
        img = m.imread(image_file)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(label_file)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        return img, lbl, os.path.basename(image_file)[:-4]

    def __len__(self):
        """Return size of dataset."""
        return len(self.image_files)

    def _glob_img_files(self):
        """Check if dataset is download and in right place."""
        self.image_files = glob.glob(os.path.join(self.img_dir, '*.jpg'))

    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """
        img = m.imresize(
            img, (self.img_size[0], self.img_size[1])
        )  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        # if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
        #     print("after det", classes, np.unique(lbl))
        #     raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


    def encode_segmap(self, mask):
        # Put all void classes to zero
        classes = np.unique(mask)
        for each_class in classes:
            assert each_class in self.class_map.keys()

        for _validc in self.class_map.keys():
            mask[mask == _validc] = self.class_map[_validc]
        return mask

def get_refuge(train, domain):
    """Get USPS dataset loader."""
    # image pre-processing
    #pre_process = transforms.Compose([transforms.ToTensor(),])

    # dataset and data loader
    refuge_dataset = REFUGE(train=train,
                        is_transform=True,
                        domain=domain)

    refuge_data_loader = torch.utils.data.DataLoader(
        dataset=refuge_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return refuge_data_loader
