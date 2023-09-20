#!/bin/bash
# DATASET_DIRS="/dataset"
# DATA_DIRS="$HOME/data"

build()
{
    docker build . -f docker/Dockerfile -t gd
}

shell() 
{
    docker run --gpus all --rm --user $(id -u):$(id -g) --shm-size=48g -it -v $(pwd):/app gd
    # docker run --gpus all --rm --user $(id -u):$(id -g) --shm-size=128g -it -v $(pwd):/app -v $DATASET_DIRS:/dataset -v $DATA_DIRS:/data gd
}

shell_root()
{
    docker run --gpus all --rm -it -v $(pwd):/app gd
    # docker run --gpus all --rm -it -v $(pwd):/app -v $DATASET_DIRS:/dataset -v $DATA_DIRS:/data gd
}

if [[ $1 == "build" ]]; then
    build
elif [[ $1 == "shell" ]]; then
    shell 
elif [[ $1 == "shell_root" ]]; then
    shell_root
fi