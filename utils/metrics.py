import numpy as np
import jax.numpy as jnp
from skimage.metrics import structural_similarity

def get_type_max(data):
    dtype = data.dtype.name
    if dtype == "uint8":
        max = 255
    elif dtype == "uint12":
        max = 4098
    elif dtype == "uint16":
        max = 65535
    elif dtype == "float32":
        max = 65535
    elif dtype == "float64":
        max = 65535
    else:
        raise NotImplementedError
    return max

def calc_ssim(gt: np.ndarray, predicted: np.ndarray):
    data_range = get_type_max(gt)
    ssim = structural_similarity(gt, predicted, data_range=data_range)
    return ssim

def calc_psnr(gt: np.ndarray, predicted: np.ndarray):
    data_range = get_type_max(gt)
    mse = jnp.mean(jnp.power(predicted / data_range - gt / data_range, 2))
    psnr = -10 * jnp.log10(mse)
    return psnr, mse

def calc_iou(gt: np.ndarray, predicted: np.ndarray):
    intersection = np.sum(gt * predicted)
    union = np.sum(gt) + np.sum(predicted) - intersection
    iou = intersection / union if union != 0 else 1.0
    return iou