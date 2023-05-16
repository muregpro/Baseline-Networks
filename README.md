# Baseline Networks for the MR to Ultrasound Registration for Prostate (µ-RegPro) Challenge

The goal of this challenge is multimodal image registration between pre-operative MR and intra-operative Ultrasound for the prostate gland. Details of the challenge are available [here](https://muregpro.github.io/). In this repository we build two simple baselines (i.e., variants of [localnet](https://www.sciencedirect.com/science/article/pii/S1361841518301051) and [voxelmorph](https://ieeexplore.ieee.org/document/8633930), with simplified backbone networks) for use with our dataset. The usage instructions are outlined below. Note that these are small, simplified networks, using resampled smaller images for training, for demonstration purposes on a wide variety of hardware.

# Usage

## Cloning the repository
```
git clone https://github.com/muregpro/Baseline-Networks.git
```

## Downloading data

The dataset may be downloaded from [this link](https://doi.org/10.5281/zenodo.7870104). For training, two directories are requried: `nifti_data/train` and `nifti_data/val` and the user may split data into these directories before training. The directory `nifti_data` must then be placed into the `Baseline-Networks` directory, such that they may be added into the docker container (see below).

## Creating a docker container

Note: `sudo` or docker group permissions may be needed to run the following commands.

1) navigate to the root directory
  ```
  cd Baseline-Networks
  ```

2) build mureg docker image from Dockerfile
```
docker build -t mureg .
```

3) create mureg docker container from mureg docker image
```
docker container create -it --name mureg mureg
```

4) start the mureg docker container
```
docker container start mureg
```


## Running commands in the docker container

```
docker exec mureg <command>
```
Examples:
```
docker exec mureg ls
```
```
docker exec mureg python3 train_localnet.py
```
```
docker exec mureg python3 train_voxelmorph.py
```
```
docker exec mureg python3 test_localnet.py
```
```
docker exec mureg python3 test_voxelmorph.py
```


## Stopping the docker container

```
docker container stop mureg
```

## Removing the docker container
```
docker container rm -f mureg
```
