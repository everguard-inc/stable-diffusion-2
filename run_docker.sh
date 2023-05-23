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
$(whoami)/stablediffusion2
