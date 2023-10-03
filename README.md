# Build and run a Docker container

```
docker build -t $(whoami)/stablediffusion .

docker run \
--rm -it \
--gpus all \
--shm-size 8G \
--hostname $(hostname) \
--workdir $(pwd) \
--user $(id -u):$(id -g) \
--mount type=bind,source="/home",target=/home \
--mount type=bind,source="/media",target=/media \
--privileged \
$(whoami)/stablediffusion
```

# Run inpainting inside the docker container

```
sh generate_using_coco.sh
```

