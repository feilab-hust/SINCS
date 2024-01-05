import numpy as np
import imageio
import tifffile
import mat73
import json
import os
import shutil
import numpy as np


def readtif(filepath, dtype=np.float32):
    """
    Read tif or tiff image stack from file.
    Returns:
        out: image stack in ndarray; (N, H, W)
    """
    # Read the tif file
    out = tifffile.imread(filepath).astype(dtype)
    return out

def group_and_readtif_folder(folder_path, step_size,threshold,normalize=True, dtype=np.float32):
    """
    Group and read tif or tiff image stacks from folder.
    Returns:
        grouped_data: a list of grouped image stacks in ndarray; [((N1,H,W), (N2,H,W), ...), ...]
        maxval_list: a list of max values for each group, used for normalizing; [float1, float2, ...]
    """
    files = os.listdir(folder_path)

    grouped_files = [files[i::step_size] for i in range(step_size)]

    grouped_data = []
    maxval_list = []
    saliency_map_list = []
    for group in grouped_files:
        group_data = []
        maxval = 0

        for filename in group:
            file_path = os.path.join(folder_path, filename)

            data = readtif(file_path, dtype)

            group_maxval = np.max(data)
            maxval = max(group_maxval, maxval)

            group_data.append(data)

        group_data_thresholded = [np.where(data >= threshold, 1, 0) for data in group_data]
        if normalize:
            group_data_normalized = [data / maxval for data in group_data]
        else:
            group_data_normalized = group_data
        grouped_data.append(tuple(group_data_normalized))
        maxval_list.append(maxval)

        # group_saliency_map = np.max(group_data_thresholded, axis=0)
        saliency_map_list.append(tuple(group_data_thresholded))

    return grouped_data, maxval_list, saliency_map_list
def load_data(path: str, datatype: str = 'multi_4D',step_size=3 ,threshold=94.1,normalize: bool = True, ret_max_val: bool = False):
    if datatype == 'multi_3D':
        images, maxval,saliency_map = load_multi_3D(path,step_size,threshold, normalize=normalize)
        out = (images) if not ret_max_val else (images, maxval,saliency_map)
    elif datatype == 'multi_4D':
        images, maxval,saliency_map = load_multi_4D(path,step_size,threshold, normalize=normalize)
        out = (images) if not ret_max_val else (images, maxval,saliency_map)
    else:
        raise NotImplementedError("Dataset type not supported!")

    return out
def load_multi_4D(path: str,step_size,threshold, normalize: bool = True):
    """
    Load default LF data.
    path: basedir path
    normalize: normalize input images by their max value or not
    """
    with open(os.path.join(path, 'transforms_train.json'), 'r') as fp:
        meta = json.load(fp)

        multi_4D_path = os.path.join(path, meta['input_path'])
        images, maxval,saliency_map = group_and_readtif_folder(multi_4D_path,step_size,threshold, normalize=normalize)



    return images,maxval,saliency_map
def load_multi_3D(path: str,step_size,threshold, normalize: bool = True):
    """
    Load default LF data.
    path: basedir path
    normalize: normalize input images by their max value or not
    """
    with open(os.path.join(path, 'transforms_train.json'), 'r') as fp:
        meta = json.load(fp)

        multi_3D_path = os.path.join(path, meta['input_path'])
        images, maxval,saliency_map = group_and_readtif_folder(multi_3D_path,step_size,threshold, normalize=normalize)



    return images,maxval,saliency_map






