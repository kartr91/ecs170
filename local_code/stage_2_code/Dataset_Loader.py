'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    # step up the stage 2 loader
    def __init__(self, data_directory):
        super().__init__('stage 2 digits', '784 pixel features (28x28), 10 classes (0-9)')
        self.data_directory = Path(data_directory)

    def load(self):
        #empty dict to hold both the training and the testing set
        data = {}
        # loop runs twice: one on train, then one on test set

        for run, file in [('train', 'train.csv'), ('test', 'test.csv')]:
            path = self.data_directory / file
            # loads rows from the saved CSV file into a NumPy array,
            # with values in a row separated by commas, and type converted to decimal(float32) for later use
            rows = np.loadtxt(path, delimiter=',', dtype=np.float32)
            # correct digit labels in col1 , remaining columns are pixel values 
            labels, pixels = rows[:, 0], rows[:, 1:]
            data[run] = {
                # scales pixels from 0-255 -> 0-1; x holds image pixels
                'X': np.ascontiguousarray(pixels / 255.0),
                # y holds correct digit labels as integers 0-9
                'y': labels.astype(np.int64),
            }
            print(f'{run}: {len(labels):,} examples, {pixels.shape[1]} features')
        return data
