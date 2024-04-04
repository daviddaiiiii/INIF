#!/bin/bash

# Create the Conda environment
conda create -n INIF python=3.10 -y
conda activate INIF

# Install JAX with CUDA support
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install the learned optimization package from GitHub
pip install git+https://github.com/google/learned_optimization.git

# Install dm-haiku from GitHub
pip install git+https://github.com/deepmind/dm-haiku

# Install other dependencies
pip install optax
pip install ipywidgets
pip install tifffile
pip install opencv-python
pip install matplotlib
pip install pandas

echo "Setup complete. The environment is ready to use."