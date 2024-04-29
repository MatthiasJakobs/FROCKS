# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""UCR Shard Descriptor."""

import logging
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import requests

from sktime.transformations.panel.rocket import Rocket
from sklearn.utils.validation import check_array
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sklearn.decomposition import PCA

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)

class UCRShardDataset(ShardDataset):
    """UCR Shard dataset class."""

    def __init__(self, x, y, data_type, rank=1, worldsize=1):
        """Initialize UCRDataset."""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        self.x = x[self.rank - 1::self.worldsize]
        self.y = y[self.rank - 1::self.worldsize]
        self.num_classes = len(np.unique(self.y))

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.x[index], self.y[index]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.x)


class UCRShardDescriptor(ShardDescriptor):
    """UCR Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            data_dir: str = '',
            kernels: int = 100,
            seed: int = 0,
            rocket: bool = False,
            **kwargs
    ):
        """Initialize UCRShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        self.data_dir = data_dir
        self.rocket = rocket
        self.kernels = int(kernels)
        self.seed = seed
        (x_train, y_train), (x_test, y_test) = self.download_data()
        self.data_by_type = {
            'train': (x_train, y_train),
            'val': (x_test, y_test)
        }

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train'):
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return UCRShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        shape = self.data_by_type['train'][0].shape
        shape_str = [str(dim) for dim in shape]
        return shape_str
        #print(str(self.data_by_type['train'][0].shape))
        #return [u" ".join(list(str(self.data_by_type['train'][0].shape)))]

    @property
    def target_shape(self):
        """Return the target shape info."""
        shape = self.data_by_type['train'][1].shape
        shape_str = [str(dim) for dim in shape]
        num_classes = str(len(np.unique(self.data_by_type['train'][1])))
        shape_str.append(num_classes)
        print(shape_str)
        return shape_str
        #print(str(self.data_by_type['train'][1].shape))
        #return [u" ".join(list(str(self.data_by_type['train'][1].shape)))]

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'UCR dataset, shard number {self.rank}'
                f' out of {self.worldsize}')
    
    def download_data(self):
        """Download prepared dataset."""
        train_df = pd.read_table("../../UCRArchive_2018/" + self.data_dir + "/" + self.data_dir + "_TRAIN.tsv", header=None)
        train_df = train_df.rename(columns={0: 'class'})
        test_df = pd.read_table("../../UCRArchive_2018/" + self.data_dir + "/" + self.data_dir + "_TEST.tsv", header=None)
        test_df = test_df.rename(columns={0: 'class'})
        
        if train_df.isna().any().any():
            train_df = train_df.dropna(axis = 1, how='any') #drop NaN by columns
            train_df = train_df.dropna(axis = 0, how='any') #drop NaN by rows
            
         
        test_df = test_df[train_df.columns.intersection(test_df.columns)] #now test set will have the same columns of train set after removing columns
        if test_df.isna().any().any():
            test_df = test_df.dropna(axis = 0, how='any') #remove missing values by rows, because columns must be the same of train set

        # Normalization
        scaler = StandardScaler()
        train_df.iloc[: , 1:] = scaler.fit_transform(train_df.iloc[: , 1:])
        test_df.iloc[: , 1:] = scaler.transform(test_df.iloc[: , 1:])
        
        x_train = train_df.iloc[: , 1:]
        x_train = x_train.values.astype(np.float32)
        x_test = test_df.iloc[: , 1:]
        x_test = x_test.values.astype(np.float32)

        label_encoder = LabelEncoder()
        # Trasformazione delle etichette sul dataframe di training
        train_df["class"] = label_encoder.fit_transform(train_df["class"])
        y_train = torch.tensor(train_df["class"].values.astype(np.float32))
        # Trasformazione delle etichette sul dataframe di test
        test_df["class"] = label_encoder.transform(test_df["class"])
        y_test = torch.tensor(test_df["class"].values.astype(np.float32))
            
        #ROCKET      
        if self.rocket == True:
            print("FROCKET: ROCKET function applied on all the data, BEFORE SHARDING")
            x_train_rocket = from_2d_array_to_nested(x_train)
            x_test_rocket = from_2d_array_to_nested(x_test)

            num_kernels = self.kernels

            rocket = Rocket(num_kernels, random_state=self.seed)

            rocket.fit(x_train_rocket)
            x_train_transform = rocket.transform(x_train_rocket)
            x_test_transform = rocket.transform(x_test_rocket)
            scaler = StandardScaler()
            X_train_transform = scaler.fit_transform(x_train_transform)
            X_test_transform = scaler.transform(x_test_transform)
            
            x_train = X_train_transform
            x_test = X_test_transform
            print("ROCKET SHAPES", x_train.shape, x_test.shape)
        
        return (x_train, y_train), (x_test, y_test)