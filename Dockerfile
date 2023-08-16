# Define the base image
FROM ubuntu:21.10

# Set environment paths
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV RETICULATE_PYTHON="~/miniconda3/envs/vambn-python/bin/python3"

# Install R and Python
RUN apt update \
    && apt install -y r-base r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev python3-dev wget

# Create a venv for reticulate
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -y -n vambn-python python=3.9

# Create a folder and copy the folder structure from local
RUN mkdir -p /vambn

COPY /vambn /vambn

# Activate the venv and install necessary libraries
RUN /bin/bash -c "cd vambn \
    && source activate vambn-python \
    && pip install -r ./02_config/requirements.txt \
    && deactivate"

# Install R libraries
RUN Rscript /vambn/02_config/install_packages.R

