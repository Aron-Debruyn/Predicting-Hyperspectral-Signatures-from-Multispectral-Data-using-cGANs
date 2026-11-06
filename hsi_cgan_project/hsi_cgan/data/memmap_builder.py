"""Building the memory-mapped training arrays.

Contents
--------
create_concatenated_memmaps  Concatenates the stored patch arrays (T_data_*.npy) into one hyperspectral and one
                             multispectral (selected bands) float32 memmap file, so training can stream from disk.
"""

import os
import numpy as np
from tqdm import tqdm


# concatenated hyperspectral and multispectral memory-mapped arrays, ensuring equal sampling
# from all input arrays while keeping the total sample count within 1000000 for MLP and Cramer architectures and 7000 in the other cases.
def create_concatenated_memmaps(input_arrays, output_dir, indices, output_channels, hyperparameters, architecture='default'):
    os.makedirs(output_dir, exist_ok=True)

    parts = output_dir.split(os.sep)
    index = parts.index("memmaps")
    path_up_to_memmaps = os.path.join(*parts[:index+1])

    hs_path = os.path.join('/'+path_up_to_memmaps, f'hyperspectral_full_{architecture}.dat')
    ms_path = os.path.join(output_dir, 'multispectral_selected_{0}.dat'.format(architecture))

    num_arrays = len(input_arrays)
    total_samples = sum(np.load(arr, mmap_mode='r').shape[0] for arr in input_arrays)
    max_samples = min(1_000_000, total_samples*250*60*60) if architecture in ['MLP', 'Cramer'] else total_samples

    samples_per_array = max_samples // num_arrays if num_arrays > 0 and architecture in ['MLP', 'Cramer'] else None
    # print('samples per array', samples_per_array) # 500000 for 2 arrays
    sample_shape = np.load(input_arrays[0], mmap_mode='r').shape[2:]  # (272, 60, 60)

    if architecture in ['MLP', 'Cramer']:
        hs_shape = (max_samples, output_channels)
        # print("hs_shape", hs_shape)
        ms_shape = (max_samples, len(indices))
        # print("ms_shape", ms_shape)
    else:
        # print("total samples: ",total_samples ," sample shape:", sample_shape)
        hs_shape = (total_samples, output_channels,) + sample_shape
        # print("hs_shape", hs_shape)
        ms_shape = (total_samples, len(indices)) + sample_shape[:]
        # print("ms_shape", ms_shape)

    hyperspectral_memmap = np.memmap(hs_path, dtype='float32', mode='w+', shape=hs_shape)
    multispectral_memmap = np.memmap(ms_path, dtype='float32', mode='w+', shape=ms_shape)

    current_idx = 0

    for arr_path in tqdm(input_arrays, desc='Processing arrays'):
        arr = np.load(arr_path, mmap_mode='r')
        # print('er voor', hyperparameters['total_length'], (hyperparameters['exclude_begin']), (hyperparameters['exclude_end']), hyperparameters['total_length']-(hyperparameters['exclude_begin']+hyperparameters['exclude_end']))
        arr = arr[:, hyperparameters['exclude_begin']:hyperparameters['total_length']-hyperparameters['exclude_end'], :, :]
        # print('er na', arr.shape)
        num_samples = arr.shape[0]

        if architecture in ['MLP', 'Cramer']:
            selected_indices = np.random.choice(num_samples*60*60, samples_per_array, replace=False)
        else:
            selected_indices = np.arange(num_samples)  # Use all samples


        if architecture in ['MLP', 'Cramer']:
            reshaped_hs = arr.transpose(0, 2, 3, 1).reshape(-1, output_channels)
            reshaped_ms = arr[:, indices, :, :].transpose(0, 2, 3, 1).reshape(-1, len(indices))

        # print('value of indices',num_samples)
        # print('the length of selected indices: ',len(selected_indices)) # 500000 for 2 arrays
        for i in selected_indices:
            if architecture in ['MLP', 'Cramer']:
                hyperspectral_memmap[current_idx] = reshaped_hs[i]
                multispectral_memmap[current_idx] = reshaped_ms[i]
            else:
                hyperspectral_memmap[current_idx] = arr[i]
                multispectral_memmap[current_idx] = arr[i, indices]

            current_idx += 1
            # print('current_idx', current_idx)
            if architecture in ['MLP', 'Cramer'] and current_idx >= max_samples:
                break

        if architecture in ['MLP', 'Cramer'] and current_idx >= max_samples:
            break
    # print('hs shape', hyperspectral_memmap.shape)
    # print('ms shape', multispectral_memmap.shape)
    hyperspectral_memmap.flush()
    multispectral_memmap.flush()

    return hs_path, ms_path
