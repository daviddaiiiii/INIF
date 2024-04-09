#!/bin/bash

# Install Tensorflow with CUDA support
pip install tensorflow==2.8.4

# Install the learned optimization package from GitHub
pip install git+https://github.com/google/learned_optimization.git

# # Install Jax
pip install -U "jax[cuda11_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html



# Install other dependencies
pip install optax
pip install ipywidgets
pip install tifffile
pip install opencv-python
pip install matplotlib
pip install pandas

echo "Setup dependencies complete. The environment is ready to use."