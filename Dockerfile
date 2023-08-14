# MIT License

# Copyright (c) 2020 FT Autonomous Team One

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# choose which SB3 image to use, CPU or GPU
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN DEBIAN_FRONTEND="noninteractive" apt-get update --fix-missing && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    python3-dev python3-pip

# Set environment variable for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install SDL2 dependencies
RUN apt-get update && apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev


RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    nano \
    git \
    unzip \
    build-essential \
    autoconf \
    libtool \
    cmake \
    vim

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade requests

RUN pip3 install \
    numpy \
    scipy \
    numba \
    Pillow \
    gym \
    pyyaml \
    pyglet==1.5.27\
    shapely \
    wandb \
    pylint

RUN pip3 install torch
RUN pip3 install stable-baselines3[extra]
# RUN pip3 install rl_zoo3

RUN apt-get install -y x11-apps
ENV HOME /home/formula
WORKDIR /home/formula/F1Tenth-RL

ENTRYPOINT ["/bin/bash"]