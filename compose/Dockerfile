# Use Nvidia Ubuntu 20 base (includes CUDA if a supported GPU is present)
# https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04@sha256:55211df43bf393d3393559d5ab53283d4ebc3943d802b04546a24f3345825bd9

# Install dependencies
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
  build-essential \
  python3.9 \
  python3.9-dev \
  python3.9-distutils \
  python3.9-venv \
  curl \
  git

# Create virtual environment and activate it
ENV VIRTUAL_ENV=/opt/venv
RUN python3.9 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install pip (we need the latest version not the standard Ubuntu version, to
# support modern wheels)
# RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && $VIRTUAL_ENV/bin/python get-pip.py


# Set python aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# User the new user
USER $USERNAME

# Install python dev dependencies
RUN pip install \
  autopep8 \
  jedi \
  mypy \
  pytest \
  toml \
  yapf \
  wget \
  gdown \
  scikit-learn \
  GPUtil

