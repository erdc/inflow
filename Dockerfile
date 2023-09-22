# syntax=docker/dockerfile:1

FROM continuumio/miniconda3

WORKDIR /usr/src/inflow

COPY . .

# The conda environment uses the following environment variables. We define
# them explicitly to make them available when running a container in
# detached mode (in which case ~/.bashrc is not executed).
ENV CONDA_DEFAULT_ENV=inflow

ENV CONDA_PREFIX=/opt/conda/envs/inflow

ENV CONDA_PREFIX_1=/opt/conda

ENV GDAL_DATA=/opt/conda/envs/inflow/share/gdal

ENV GDAL_DRIVER_PATH=/opt/conda/envs/inflow/lib/gdalplugins

ENV PROJ_DATA=/opt/conda/envs/inflow/share/proj

ENV PROJ_NETWORK=ON

# Define arguments for convenience.
ARG CONDA_ENV_YAML=inflow_environment.yml

ARG CONDA_INIT_PATH=/opt/conda/etc/profile.d/conda.sh

# Change the default shell to bash to execute the commands in the RUN
# instruction below.
SHELL ["/bin/bash", "-c"]

RUN source $CONDA_INIT_PATH \
    && conda env create -f $CONDA_ENV_YAML \
    && conda activate $CONDA_DEFAULT_ENV \
    && pip install . \
    && echo "conda activate $CONDA_DEFAULT_ENV" >> ~/.bashrc

# Prepend conda and conda environment paths to the PATH environment variable.
ENV PATH=$CONDA_PREFIX/bin:$CONDA_PREFIX_1/condabin:$PATH

