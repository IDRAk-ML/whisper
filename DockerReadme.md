# Docker Readme

If not setup do setup it first
Setup Nvidia Docker
```
sudo apt-get update && sudo apt-get install -y \
    curl \
    ca-certificates \
    gnupg \
    lsb-release

curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/nvidia-container.gpg

distribution=ubuntu22.04

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#http://#https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker


docker run --rm --gpus all nvidia/cuda:12.6.2-runtime-ubuntu22.04 nvidia-smi
```

1. Run like this.
```
docker build -t whisper-fastapi .
docker run --rm --gpus all -p 9005:9005 -v $(pwd)/temp:/workspace/temp whisper-fastapi
```

