import os
import time
import numpy as np
from utils import others
import subprocess

def calc_bitrate(data, compressed_bitnum, fps=30):
    fnum = data.shape[0]
    return compressed_bitnum / fnum * fps / 1000 # not 1024

def compress(data, meta_data, compressed_data_path = '../save/test.mp4', test_only=True):
    if data.ndim != 3:
        raise ValueError("Data should be 3D for HEVC compression")
    data = np.expand_dims(data, axis=-1)
    # Padding the raw data
    ori_shape = data.shape
    padded_original_data = np.zeros([
        int(ori_shape[0]),
        int(np.ceil(ori_shape[1] / 2) * 2),
        int(np.ceil(ori_shape[2] / 2) * 2),
        int(ori_shape[3]),
    ]).astype(data.dtype)
    print(f'Padded image shape {padded_original_data.shape}')
    padded_original_data[:ori_shape[0], :ori_shape[1], :ori_shape[2], :ori_shape[3]] = data

    # Calculate the bytes of the data
    orig_data_bytes = (meta_data['size'] / np.prod(ori_shape) * np.prod(padded_original_data.shape))
    
    if padded_original_data.dtype == np.uint8:
        pix_fmt = "gray8"
    elif padded_original_data.dtype == np.uint16:
        pix_fmt = "gray16"
    elif padded_original_data.dtype == np.float32:
        pix_fmt = "grayf32le"
    else:
        raise NotImplementedError

    coder = "libx265"
    h, w = padded_original_data.shape[1:3]
    bitrate = calc_bitrate(padded_original_data, orig_data_bytes * 8 / round(meta_data['ratio']), fps=30)

    # Compression
    command = [
        "ffmpeg", "-y", "-loglevel", "error",  # 添加这个来减少输出
        "-f", "rawvideo", "-s", "%dx%d" % (w, h),
        "-pix_fmt", pix_fmt, "-r", "31", "-i", "-",
        "-an", "-c:v", coder, "-b:v", "%dk" % (bitrate),
        "-x265-params", f"crf={meta_data['crf']}:bframes=0:aq-mode=3", "-preset", "medium", compressed_data_path,
    ]
    with open(os.devnull, 'w') as fnull: # delete the process info
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=fnull)
        pipe.communicate(padded_original_data.tobytes())
        pipe.wait()
    
    HEVC_bytes = os.path.getsize(compressed_data_path)
    actual_ratio = meta_data["size"] / HEVC_bytes
    print("Compressed Done!", flush=True)
    print(f'HEVC CRF:{meta_data["crf"]}, compressed bpp:{bitrate}')
    print(f'Original_MB:{others.bytes_to_MB(meta_data["size"])}MB')
    print(f'compressed_MB:{others.bytes_to_MB(HEVC_bytes)}MB')
    print(f'target ratio : {meta_data["ratio"]}')
    print(f'actual ratio : {actual_ratio}')
    meta_data['HEVC_bytes'] = HEVC_bytes
    meta_data['pad_shape'] = padded_original_data.shape
    meta_data['pix_fmt'] = pix_fmt
    # delete the process info
    if test_only:
        os.remove(compressed_data_path)
        print(f"Remove {compressed_data_path}")
        return actual_ratio

def decompress(compressed_data_path, meta_data):
    bit_type = meta_data['pix_fmt']
    shape = meta_data['pad_shape']
    ori_shape = tuple(meta_data['shape'].values())
    # Decompress
    command = [
        "ffmpeg", "-i", compressed_data_path, "-f", "image2pipe",
        "-pix_fmt", bit_type, "-an", "-vcodec", "rawvideo", "-",
    ]
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**9)

    if bit_type == 'gray8':
        image = pipe.stdout.read(np.prod(shape))
    elif bit_type == 'gray16':
        image = pipe.stdout.read(np.prod(shape) * 2)
    elif bit_type == 'grayf32le':
        image = pipe.stdout.read(np.prod(shape) * 4)
    else:
        raise NotImplementedError

    decompressed_data = np.frombuffer(image, dtype=meta_data['dtype']).reshape(shape)
    pipe.stdout.flush()
    time_e = time.time()
    decompressed_data = decompressed_data[:ori_shape[0], :ori_shape[1], :ori_shape[2], :].squeeze()
    return decompressed_data