# Baseline Networks for the MR to Ultrasound Registration for Prostate (µ-RegPro) Challenge

# Usage

## Creating a docker container for µ-reg

1) navigate to root directory
  ```
  cd mureg
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
docker start container mureg
```


## Running commands in the docker container

```
docker exec mureg <command>
```
Examples:
```
docker exec mureg ls mureg
```
```
docker exec mureg python3 mureg/train_localnet.py
```
```
docker exec mureg python3 mureg/train_voxelmorph.py
```
```
docker exec mureg python3 mureg/test_localnet.py
```
```
docker exec mureg python3 mureg/test_voxelmorph.py
```


## Stopping the docker container

```
docker container stop mureg
```
