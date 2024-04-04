import sys
import tifffile
import numpy as np
import cv2
import jax.numpy as jnp
import jax
import os
import pickle
from utils import visualizer

def norm (data, meta_data:dict):
    return data / (meta_data['max'] - meta_data['min']) * 100

def denorm (data, meta_data:dict):
    return data * (meta_data['max'] - meta_data['min'])  / 100

def get_shape_dict(raw_data: np.ndarray) -> dict[str, int]:
    '''
    this function gets the shape of the data and the order of the dimensions in the data.
    
    input -> raw_data - the data
    
    output -> shape_dict - key: dimension, value: size of the dimension
    '''
    dimension_order = input(f"Current shape {raw_data.shape}, enter the dimension order of the data, using ZCTXY or zctxy (defualt only assign xy to the end): ")
    shape_dict = {}
    if dimension_order == "":
        for i, d in enumerate(raw_data.shape[:-2]):
            shape_dict[f"dim{i}"] = d
        shape_dict['X'] = raw_data.shape[-2]
        shape_dict['Y'] = raw_data.shape[-1]
    for i, d in enumerate(dimension_order):
        shape_dict[d] = raw_data.shape[i]
    return shape_dict

def bytes_to_MB(size: int) -> float:
    '''
    this function converts bytes to MB
    
    input -> size in bytes
    
    output -> size in MB
    '''
    return size / 1024 / 1024

def read_data(raw_data_path, meta_data: dict) -> np.ndarray:
    '''
    this function reads the data from the file, record the meta, and normalize the data.
    
    output -> normalized data
    '''
    print("The path to the file is: ", raw_data_path)
    # read the file (support TIF, TIFF, NPY, NPZ, PNG)
    if raw_data_path.endswith(".tif") or raw_data_path.endswith(".tiff"):
        raw_data = tifffile.imread(raw_data_path)
    elif raw_data_path.endswith(".npy"):
        raw_data = np.load(raw_data_path)
    elif raw_data_path.endswith(".npz"):
        raw_data = np.load(raw_data_path)['data']
    elif raw_data_path.endswith(".png"):
        raw_data = cv2.imread(raw_data_path, cv2.IMREAD_UNCHANGED)
    else:
        print("Unsupported file format. Please use TIF, TIFF, NPY, NPZ, or PNG.")
        exit()
    # get shape of the data
    print(f"The shape of the raw data is: {raw_data.shape}")
    # get size of the data
    raw_data_size = os.path.getsize(raw_data_path)
    print(f"The size of the raw data is: {bytes_to_MB(raw_data_size)} MB")
    meta_data['name'] = raw_data_path.split('/')[-1].split('.')[0]
    meta_data['size'] = raw_data_size
    meta_data['shape'] = raw_data.size
    return raw_data

def preprocess(ROI_data, meta_data: dict) -> np.ndarray:
    shape_dict = get_shape_dict(ROI_data)
    print(f"The shape of the ROI data is: {shape_dict}")
    # get size of the data
    ROI_size = meta_data['size']*ROI_data.size/meta_data['shape']
    print(f"The size of the ROI data is: {bytes_to_MB(ROI_size)} MB")
    # get data type of the data
    print(f"The data type of the raw data is: {ROI_data.dtype}")
    # get max and min values of the data
    print(f"The max value of the raw data is: {ROI_data.max()}, and the min value is: {ROI_data.min()}")
    meta_data['shape'] = shape_dict
    meta_data['size'] = ROI_size
    meta_data['dtype'] = ROI_data.dtype
    meta_data['max'] = ROI_data.max()
    meta_data['min'] = ROI_data.min()
    
    ROI_data = norm(ROI_data, meta_data)
    return ROI_data


class Sampler:
    def __init__(self, data, meta_data) -> None:
        '''
        this function initializes the random sampler
        
        input -> data
        '''
        self.data = data
        self.batch_size = meta_data['batch_size']

    def next(self, batch_size = None) -> tuple:
        '''
        This function creates random coordinates and extracts the corresponding
        values from the data without explicitly generating full coordinate arrays.

        input -> data - the multidimensional data array
                num_samples - number of random samples to generate

        output -> random_coords - random normalized coordinates
                values - the values from the data at the random coordinates
        '''
        if batch_size is not None:
            self.batch_size = batch_size
        shape = self.data.shape
        # randomly sample indices for each dimension
        random_indices = tuple(np.random.randint(0, s, self.batch_size) for s in shape)
        # extract the values at the random indices
        values = self.data[random_indices]
        # calculate the coordinates and normalize them to [-1, 1]
        random_coords = jnp.vstack([(indices / (s - 1)) * 2 - 1 for indices, s in zip(random_indices, shape)]).T
        return random_coords, values
    

def save(meta_data, params = None, dir = '../save/'):
    '''
    this function saves the file
    
    input -> params: the parameters of the model
            meta_data: the metadata of the data
            dir: the directory to save the file
    '''
    if params is None:
        with open(os.path.join(dir, 'meta_data.pkl'), 'wb') as f:
            pickle.dump(meta_data, f)
            print(f'saved at {os.path.join(dir, "meta_data.pkl")}')
    else:
        model_name = meta_data['model']['name']
        file_name = f'{model_name}_name_{meta_data["name"]}_ratio_{meta_data["ratio"]}_loop_num_{meta_data["loop_number"]: 4f}.pkl'
        file_name = os.path.join(dir, file_name)
        with open(file_name, 'wb') as f:
            pickle.dump((params, meta_data), f)
        print(f'saved at {file_name}')


def load(file_name):
    '''
    this function loads the file
    
    input -> file_name: the name of the file
    
    output -> params: the parameters of the model
            meta_data: the metadata of the data
    '''
    with open(file_name, 'rb') as f:
        params, meta_data = pickle.load(f)
    return params, meta_data