# INIF
This repo is the offical codebase for the manuscript 'Implicit Neural Image Field for Biological Microscopy Image Compression'.

## Introduction
**Implicit neural image field (INIF)** is a SOTA paradigm for microscopy image compression, characterized by its effectiveness and flexibility. It adopts a learned optimizer and leverages common codecs for faster compression, while integrating application-specific guidance for improved compression quality and trustworthiness.
## System Requirements

### Operating System
> [!NOTE] 
> Currently only **Linux System** is supported.

### Dependencies

<details>
<summary>Framework</summary>

- jax==0.4.26[cuda12_pip]
- tensorflow==2.8.4 (Nvidia-GPU version)

</details>

<details>
<summary>pip</summary>

- dm-haiku
- learned optimizer (lopt)
- optax
- ipywidgets
- tifffile
- opencv-python
- matplotlib
- pandas
- learned-optimization

</details>

## Installation

### From source:
   1. clone this repository:
      ```bash
      git clone https://github.com/daviddaiiiii/INIF.git
      ```
   2. Create the conda environment and activate:
      ```bash
      conda deactivate
      conda create -n INIF python=3.10
      conda activate INIF
      ```
   3. Run the `setup.sh` file to install automatically: 
      ```bash
      chmod +x setup.sh
      ./setup.sh
      ```
      These codes would:
         1. Give the script execution permissions
         2. Install the [dependencies](#dependencies) via `pip`

> [!TIP] 
> In case of issues when downloading the weight of learned optimizers automatically from Google Cloud, below are some workaround:
<details>
<summary>Some workaround:</summary>

   1. We provide the default weight and continue trained weight in '../INIF/Learned_optimizer_weight'
   2. go to '../anaconda3/envs/INIF/lib/pythonX.XX/site-packages/learned_optimization/research/general_lopt/pretrained_optimizers.py'

      > :bulb: 
      > `ctrl`+`click` on 'from learned_optimization.research.general_lopt import pretrained_optimizers' to jump to the file

   3. replace the path in line143 to '../INIF/Learned_optimizer_weight
      ```bash
      _pretrain_no_config_root = 'gs://gresearch/learned_optimization/pretrained_lopts/no_config/'
      ```

</details>

### Docker:

1. Install docker if you don't have it: [get-docker](https://docs.docker.com/get-docker/)
> [!IMPORTANT]  
> To utilize GPU, [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt) package is also required.


2. retrieve the docker image from the dockerhub:
   ```bash
   docker pull foo/bar
   ```
3. start a container:
   ```bash
   docker run --gpus all -it --rm --shm-size 16G --ulimit memlock=-1 -v ./INIF:/INIF/ --name inif inif
   ```

   <details>

   <summary>where:</summary>

   - `--gpus`: use the gpu
   - `-it`: interact with the container
   - `--rm`: remove the container after exit
   - `--shm-size`: set the shared memory size to avoid memory issues
   - `--ulimit memlock=-1`: remove mem lock limit
   - `-v`: mount the current directory to the container
   - `--name`: name the container
   
   </details>

## Quick Start :rocket:
We provide a **google colab notework** that demonstrates how to use INIF for the general image compression: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daviddaiiiii/INIF/blob/main/compression_tutorial.ipynb)