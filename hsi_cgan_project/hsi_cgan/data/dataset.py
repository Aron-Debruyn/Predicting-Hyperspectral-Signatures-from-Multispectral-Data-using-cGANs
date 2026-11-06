"""Dataset used to stream the memmapped data during training and evaluation.

Contents
--------
HyperspectralDataset  IterableDataset over the hyperspectral / multispectral memmaps. Provides the fixed
                      train / val / test split (by image, 53 patches per image), per-worker sharding,
                      shuffling and batching. Yields (multispectral_batch, hyperspectral_batch).
"""

import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class HyperspectralDataset(IterableDataset):
    def __init__(self, hyperspectral_path, multispectral_path, batch_size, split, output_channels, indices,
                 data_percentage=1.0, random_seed=42, architecture='default'):
        self.output_channels = output_channels
        self.indices = indices

        self.hyperspectral_memmap = np.memmap(
            hyperspectral_path, dtype='float32', mode='r',
            shape=self._get_memmap_shape(hyperspectral_path, architecture)
        )
        self.multispectral_memmap = np.memmap(
            multispectral_path, dtype='float32', mode='r',
            shape=self._get_memmap_shape_ms(multispectral_path, architecture)
        )

        self.architecture = architecture
        self.batch_size = min(batch_size, 1024) if architecture in ['MLP', 'Cramer'] else batch_size
        
        assert len(self.hyperspectral_memmap) == len(self.multispectral_memmap), \
            "Hyperspectral and multispectral data must have same number of samples"

        self.data_percentage = data_percentage
        self.split = split
        self.random_seed = random_seed

        patches_per_image = 53 # Hardcoded for now
        total_images = len(self.hyperspectral_memmap) // patches_per_image
        
        # Define clean image-level counts (approx 70-20-10 split of 133 images)
        num_train_images = 93  # 93 * 53 = 4,929 samples
        num_val_images = 27    # 27 * 53 = 1,431 samples
        # Remaining 13 images go to test = 689 samples
        
        train_end = num_train_images * patches_per_image
        val_end = (num_train_images + num_val_images) * patches_per_image

        self.split_ranges = {
            'train': (0, train_end),
            'val': (train_end, val_end),
            'test': (val_end, len(self.hyperspectral_memmap))
        }


        self.start_idx, self.end_idx = self.split_ranges[split]
        self.num_samples = int((self.end_idx - self.start_idx) * data_percentage)

        self.hyperspectral_data = self.hyperspectral_memmap[self.start_idx:self.end_idx]
        self.multispectral_data = self.multispectral_memmap[self.start_idx:self.end_idx]

    def _get_memmap_shape(self, path, architecture):
        file_size = os.path.getsize(path)

        if architecture in ['MLP', 'Cramer']:

            # print('file size HS', (file_size // (4 * self.output_channels), self.output_channels))
            return (file_size // (4 * self.output_channels), self.output_channels)  # 4 bytes/float32
        else:
            # print('file size HS', (file_size // (4 * self.output_channels * 60 * 60), self.output_channels, 60, 60))
            return (file_size // (4 * self.output_channels * 60 * 60), self.output_channels, 60, 60)


    def _get_memmap_shape_ms(self, path, architecture):
        file_size = os.path.getsize(path)
        if architecture in ['MLP', 'Cramer']:

            # print('file size MS', (file_size // (4 * len(self.indices)), len(self.indices)))
            return (file_size // (4 * len(self.indices)), len(self.indices))
        else:
            # print('file size MS', (file_size // (4 * len(self.indices) * 60 * 60), len(self.indices), 60, 60))
            return (file_size // (4 * len(self.indices) * 60 * 60), len(self.indices), 60, 60)



    def __iter__(self):

        worker_info = get_worker_info()

        if worker_info is None:
            start = self.start_idx
            end = self.start_idx + self.num_samples
        else:
            # Split the data across workers
            per_worker = int(self.num_samples / worker_info.num_workers)
            worker_id = worker_info.id
            start = self.start_idx + worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else self.start_idx + self.num_samples


        num_batches = (end - start) // self.batch_size


        np.random.seed(self.random_seed)


        indices = np.arange(start, end)
        np.random.shuffle(indices)


        if self.architecture in ['MLP', 'Cramer']:
            # print("Hyperspectral batch shape:", hyperspectral_batch.shape)
            # print("Multispectral batch shape:", multispectral_batch.shape)

            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_indices = indices[batch_start:batch_end]

                # Shape will be [batch_size, 20] for multispectral
                multispectral_batch = self.multispectral_memmap[batch_indices]
                hyperspectral_batch = self.hyperspectral_memmap[batch_indices]

                # print("Hyperspectral batch shape:", hyperspectral_batch.shape)
                # print("Multispectral batch shape:", multispectral_batch.shape)
                yield (
                    torch.tensor(multispectral_batch, dtype=torch.float32),
                    torch.tensor(hyperspectral_batch, dtype=torch.float32)
                )
        else:


            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_indices = indices[batch_start:batch_end]


                # hyperspectral_batch = self.hyperspectral_data[batch_indices] #Adjusted these two lines
                # multispectral_batch = self.multispectral_data[batch_indices]
                hyperspectral_batch = self.hyperspectral_memmap[batch_indices]
                multispectral_batch = self.multispectral_memmap[batch_indices]

                # print("Number of zeros in hyperspectral_batch:", np.count_nonzero(hyperspectral_batch == 0))
                # print("Hyperspectral batch shape:", hyperspectral_batch.shape)
                # print("Multispectral batch shape:", multispectral_batch.shape)


                hyperspectral_batch = torch.tensor(hyperspectral_batch, dtype=torch.float32)
                multispectral_batch = torch.tensor(multispectral_batch, dtype=torch.float32)

                yield multispectral_batch, hyperspectral_batch

    def __len__(self):

        num_batches = self.num_samples // self.batch_size
        return num_batches
