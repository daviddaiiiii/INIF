# INIF
Code for 'Implicit Neural Image Field for Biological Microscopy Image Compression' 

## Installation
You can use the setup.sh file to install automatically in linux. 
```bash
chmod +x setup.sh
./setup.sh
```
These codes would
1. Give the script execution permissions.
2. Execute follow commands
### (Note: You can execute manually if you have a environment which already contains most of the dependencies)
### Conda Environment
```bash
conda create -n INIF python=3.10
conda activate INIF
```
### Frameworks
JAX (Nvidia-GPU version)
```bash
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
### PIP
learned optimizer (lopt)
```bash
 pip install git+https://github.com/google/learned_optimization.git
```
dm-haiku
```bash
pip install git+https://github.com/deepmind/dm-haiku
```
optax
```bash
pip install optax
```
ipywidgets 
```bash
pip install ipywidgets
```
tifffile
```bash
pip install tifffile
```
opencv-python
```bash
pip install opencv-python
```
matplotlib
```bash
pip install opencv-python matplotlib
```
pandas
```bash
pip install pandas
```
### (Note: If you facing issues when downloading the weight of learned optimizers automatically from Google Cloud, here are the instructions)
1. We provide the default weight and tuned weight in '../INIF/Learned_optimizer_weight'
2. 
## Demonstration
