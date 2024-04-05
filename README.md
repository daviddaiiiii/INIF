# INIF
Code for 'Implicit Neural Image Field for Biological Microscopy Image Compression' 

## Installation
First clone the repository by
```bash
git clone https://github.com/daviddaiiiii/INIF.git
```
Create the conda environment
```bash
conda deactivate
conda create -n INIF python=3.10
conda activate INIF
```
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
1. We provide the default weight and continue trained weight in '../INIF/Learned_optimizer_weight'
2. go to '../anaconda3/envs/INIF/lib/pythonX.XX/site-packages/learned_optimization/research/general_lopt/pretrained_optimizers.py'

   (tips: ctrl+click on 'from learned_optimization.research.general_lopt import pretrained_optimizers' to jump to the file)

3. replace the path in line143 to '../INIF/Learned_optimizer_weight
```bash
 _pretrain_no_config_root = 'gs://gresearch/learned_optimization/pretrained_lopts/no_config/'
```
## Demonstration
