# Baseline Networks for the MR to Ultrasound Registration for Prostate (µ-RegPro) Challenge

# Usage

## Cloning the repository
```
git clone https://github.com/muregpro/Baseline-Networks.git
```

## Creating a docker container for µ-reg

1) navigate to root directory
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
