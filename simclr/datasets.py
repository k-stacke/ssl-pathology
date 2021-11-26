import random
from PIL import Image
import lmdb
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms

class ImagePatchesDataset(Dataset):
    def __init__(self, opt, dataframe, image_dir, transform=None, label_enum=None):
        self.opt = opt
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = opt.image_size

        self.label_enum = {'TUMOR': 1, 'NONTUMOR': 0} if label_enum is None else label_enum
        print(self.label_enum)

    def __len__(self):
        return len(self.dataframe.index)


    def get_views(self, image):
        pos_1 = self.transform(image)
        pos_2 = torch.zeros_like(pos_1) if self.opt.train_supervised else self.transform(image)
        return pos_1, pos_2


    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        path = f"{self.image_dir}/{row.filename}"
        try:
            image = Image.open(path) # pil image
        except IOError:
            print(f"could not open {path}")
            return None

        pos_1, pos_2 = self.get_views(image)

        label = self.label_enum[row.label]

        try:
            id_ = row.patch_id
        except AttributeError:
            id_ = row.filename
        return pos_1, pos_2, label, id_, row.slide_id


class LmdbDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path, transform):
        self.cursor_access = False
        self.lmdb_path = lmdb_path
        self.image_dimensions = (224, 224, 3) # size of files in lmdb
        self.transform = transform

        self._init_db()

    def __len__(self):
        return self.length

    def _init_db(self):
        num_readers = 999

        self.env = lmdb.open(self.lmdb_path,
                             max_readers=num_readers,
                             readonly=1,
                             lock=0,
                             readahead=0,
                             meminit=0)

        self.txn = self.env.begin(write=False)
        self.cursor = self.txn.cursor()

        self.length = self.txn.stat()['entries']
        print('Generating keys to lmdb dataset, this takes a while...')
        self.keys = [key for key, _ in self.txn.cursor()] # not so fast...

    def close(self):
        self.env.close()

    def __getitem__(self, index):
        ' cursor in lmdb is much faster than random access '
        if self.cursor_access:
            if not self.cursor.next():
                self.cursor.first()
            image = self.cursor.value()
        else:
            image = self.txn.get(self.keys[index])

        image = np.frombuffer(image, dtype=np.uint8)
        image = image.reshape(self.image_dimensions)
        image = Image.fromarray(image)

        pos_1 = self.transform(image)
        pos_2 = self.transform(image)

        return pos_1, pos_2

