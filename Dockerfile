FROM --platform=linux/amd64 ubuntu:24.04

RUN apt update && apt install -y make rsync git vim
RUN apt install -y python3.12-venv

#Download necessary libraries
RUN apt-get update && apt-get -y upgrade && apt-get install -y \
  build-essential \
  byobu \
  curl \
  git \
  htop \
  man \
  unzip \
  vim \
  wget \
  openssh-client \
  gfortran \
  mpich \
  libmpich-dev \
  libfftw3-dev \
  python3 \
  python-is-python3 \
  openmpi-bin \
  openmpi-doc \
  libopenmpi-dev \
  libopenblas-dev \
  python3-pip

#Download s5cmd as per recommended installation (see https://github.com/Rose-STL-Lab/Zihao-s-Toolbox)
RUN wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_linux_amd64.deb && dpkg -i s5cmd_2.2.2_linux_amd64.deb && rm s5cmd_2.2.2_linux_amd64.deb

#Clean up any installation caches after apt installs, make directories for simulation runs
RUN rm -rf /var/lib/apt/lists/* && \
    mkdir /home/user && \
    mkdir /home/user/sim

WORKDIR /home/user

# Download and install Miniconda
RUN cd ../ && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh

# Add Conda to the PATH
ENV PATH="/usr/local/bin:${PATH}"

# Accept Conda TOS 
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Update Conda and clean up
RUN conda update -y conda && \
    conda clean --all --yes

WORKDIR /
# Copy over private key, and set permissions
# Warning! Anyone who gets their hands on this image will be able
# to retrieve this private key file from the corresponding image layer
RUN mkdir -p /root/.ssh
ADD /.ssh/id_rsa /root/.ssh/id_rsa
ADD /.ssh/known_hosts /root/.ssh/known_hosts

# Create known_hosts
RUN ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts

# Remove host checking
RUN echo "Host github.com\n\tStrictHostKeyChecking no\n" >> /root/.ssh/config

# Update permissions
RUN chmod 400 /root/.ssh/id_rsa
RUN chmod 400 /root/.ssh/config
RUN chmod 400 /root/.ssh/known_hosts

WORKDIR /home/user/

# Clone this repo and set up conda env (requires conda install)
# Only cloning necessary submodules for cgyro runs due to SSH permission issues for other submodules
RUN git clone --depth=1 --branch cgyro git@github.com:zclawr/costsci-tools.git && \
    cd ./costsci-tools && \
    git pull && \
    git submodule update --init --recursive

WORKDIR /home/user/costsci-tools

# Install conda environment
RUN conda update --all
RUN conda env create -n costsci-tools --file environment.yml
RUN conda clean -qafy

# Activate the new conda environment and install pyyaml
SHELL ["/opt/conda/bin/conda", "run", "-n", "costsci-tools", "/bin/bash", "-c"]
RUN pip install pyyaml

# Uncomment this if you want to test the docker container outside of kube
# ENTRYPOINT ["sleep", "infinity"]