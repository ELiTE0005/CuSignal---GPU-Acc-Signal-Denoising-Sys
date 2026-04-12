# Specify the base NVIDIA container with RAPIDS & CUDA
FROM rapidsai/base:24.04-cuda12.2-py3.10

USER root

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER rapids

# Install additional python dependencies (h5py, cusignal) and visualization libraries
RUN mamba install -y -c conda-forge -c nvidia \
    cusignal \
    cudf \
    cuml \
    cugraph \
    h5py \
    matplotlib \
    pytest \
    scipy \
    tqdm \
    websockets \
    && mamba clean -ya

# Set the working directory
WORKDIR /app/cusignal_project

# Keep container running
CMD ["sleep", "infinity"]
