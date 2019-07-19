__all__ = ['MultiLabelDataset']


import os
import io
import logging
import torch
from PIL import Image, ImageFile
from .base_dataset import BaseDataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MultiLabelDataset(BaseDataset):

    def __init__(self, imglist, root, reader, transform, num_classes,
                 maxlen=None, dummy_read=False, dummy_size=None, **kwargs):
        super(MultiLabelDataset, self).__init__(**kwargs)

        self.root = root
        self.reader = reader
        self.transform = transform
        self.num_classes = num_classes
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        if dummy_read and dummy_size is None:
            raise ValueError('if dummy_read is True, should provide dummy_size')

        self.imglist = []
        with open(imglist) as f:
            for line in f.readlines():
                path, label = line.strip().split(maxsplit=1)
                label = [int(l) for l in label.split()]
                self.imglist.append((path, label))

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        image_name, label = self.imglist[index]
        if self.root != '' and image_name.startswith('/'):
            raise RuntimeError('root not empty but image_name starts with "/"')
        path = os.path.join(self.root, image_name)

        dense_label = torch.LongTensor(label)
        onehot_label = torch.zeros(self.num_classes)
        onehot_label.scatter_(0, dense_label, 1)
        sample = {'label': onehot_label}
        try:
            if not self.dummy_read:
                filebytes = self.reader(path)
                buff = io.BytesIO(filebytes)
            if self.dummy_size is not None:
                sample['data'] = torch.rand(self.dummy_size)
            else:
                image = Image.open(buff)
                sample['data'] = self.transform(image)
        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e

        return sample
