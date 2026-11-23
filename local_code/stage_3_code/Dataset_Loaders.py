"""loading classes for the three datasets"""""

import os
import pickle

import numpy as np

from local_code.base_class.dataset import dataset


class Image_Dataset_Loader(dataset):
    def __init__(self, dir, name, channels, label_offset=0):
        super().__init__(name, '')
        self.dir = dir
        self.channels = channels
        self.offset = label_offset

    def load(self):
        # load pickled dataset file from data/stage_3_data
        file = open(os.path.join(self.dir, self.dataset_name), 'rb')
        saved_data = pickle.load(file)
        file.close()

        # make image/label arrays for train/test
        data = {}
        for split in ('train', 'test'):
            examples = saved_data[split]
            # dynamically gets height/width of the first image
            height, width = np.asarray(examples[0]['image']).shape[:2]
            # np array arranged as img #,  channel, h, w
            # e.g. (6000, 1, 28, 28) for MNIST
            images = np.empty((len(examples), self.channels, height, width), dtype=np.float32)
            labels = np.empty(len(examples), dtype=np.int64)

            for index, example in enumerate(examples):
                image = np.asarray(example['image'])
                if self.dataset_name == 'ORL':
                    # 3 channels have same val
                    images[index, 0] = image[:, :, 0]
                elif self.channels == 1:
                    images[index, 0] = image
                else:
                    images[index] = image.transpose(2, 0, 1)
                labels[index] = example['label'] - self.offset

            # scale pixel vals
            images /= 255.0
            data[split] = {'X': images, 'y': labels}
            print(f'{self.dataset_name} {split}: {len(examples):,} images, {images.shape[1:]}')
        return data

# dataset-specific loaders with their respective channels/offset
class MINSTLoader(Image_Dataset_Loader):
    def __init__(self, dir):
        super().__init__(dir, 'MNIST', channels=1)


class ORLLoader(Image_Dataset_Loader):
    def __init__(self, dir):
        super().__init__(dir, 'ORL', channels=1, label_offset=1)


class CIFARLoader(Image_Dataset_Loader):
    def __init__(self, dir):
        super().__init__(dir, 'CIFAR', channels=3)
