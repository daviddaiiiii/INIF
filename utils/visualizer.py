import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
import pandas as pd
from IPython.display import display


class DataVisualizer:
    def __init__(self, data, shape_dict):
        self.data = data
        self.shape_dict = shape_dict
        self.current_indices = None
        self.sliders = {}
        self.setup_ui()
    
    def setup_ui(self):
        for dim, size in self.shape_dict.items():
            if dim not in ['X', 'Y', 'x', 'y']:
                self.sliders[dim] = widgets.IntSlider(
                    value=size//2,
                    min=0,
                    max=size - 1,
                    step=1,
                    description=dim,
                    continuous_update=False
                )
        ui = widgets.HBox([self.sliders[dim] for dim in self.sliders])
        out = widgets.interactive_output(self.multi_dim_slice, self.sliders)
        display(ui, out)
    
    def multi_dim_slice(self, **kwargs):
        self.current_indices = tuple(kwargs.get(dim, slice(0, self.shape_dict[dim], 1)) for dim in self.shape_dict.keys())
        print(self.current_indices)
        sliced_data = self.data[self.current_indices]
        self.plot2d(sliced_data)
    
    def get_indices(self):
        return self.current_indices
    
    def plot2d(self, sliced_data):
        return plot2d(sliced_data)

def plot2d(data: np.ndarray, size:int = 6, cmap:str = 'gray'):
    '''
    this function plots the 2D data
    
    input -> data: the 2D data
            size: the size of the plot
            cmap: the color map to use ('gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', etc.)
    '''
    if data.ndim > 2:
        try :
            data = data.squeeze()
        except:
            raise ValueError("Data has more than 2 dimensions")
    plt.figure(figsize=(size, size))
    plt.imshow(data, cmap=cmap)
    plt.axis('off')
    plt.show()


def get_slice(meta_data, slice_indices: tuple,  downsample_ratio: int = 1):
    shape = tuple(meta_data['shape'].values())
    # Initialize the coordinate grid for the last two dimensions
    coord_grid_x, coord_grid_y = np.meshgrid(
        np.arange(slice_indices[-2].start, slice_indices[-2].stop, downsample_ratio),
        np.arange(slice_indices[-2].start,  slice_indices[-1].stop, downsample_ratio),
        indexing='ij'
    )
    # Normalize the coordinates to the range [-1, 1]
    coord_grid_x = (coord_grid_x / (shape[-2] - 1)) * 2 - 1
    coord_grid_y = (coord_grid_y / (shape[-1] - 1)) * 2 - 1
    image_shape = (coord_grid_x.shape)
    # Flatten the coordinate grid and combine x and y coordinates
    slice_coords = np.vstack((coord_grid_x.flatten(), coord_grid_y.flatten())).T
    # Prepend fixed normalized coordinates for the preceding dimensions
    preceding_coords = [
        ((s / (shape[dim] - 1)) * 2 - 1) if isinstance(s, int) else 0
        for dim, s in enumerate(slice_indices[:-2])
    ]
    # Repeat preceding_coords to match the number of rows in slice_coords
    preceding_coords = np.repeat(
        np.array(preceding_coords)[np.newaxis, :], 
        slice_coords.shape[0], 
        axis=0
    )
    # Concatenate the fixed coordinates with the downsampled coordinates
    slice_coords = np.hstack((preceding_coords, slice_coords))

    return slice_coords, image_shape

def display_metadata(meta_data, size = (8,4)):
    df = pd.DataFrame(list(meta_data.items()), columns=['Key', 'Value'])
    fig, ax = plt.subplots(figsize=size)
    # ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center') # type: ignore
    plt.show()