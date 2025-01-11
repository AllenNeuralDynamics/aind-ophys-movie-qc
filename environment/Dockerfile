# hash:sha256:5089fb701b3116323ba2655f060ee2923c3b31045a29630498987e95890960f7
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PIPELINE_URL="https://codeocean.allenneuraldynamics.org/capsule/6761117/tree"
ENV PIPELINE_VERSION="2.0"
ENV VERSION="3.0"

ENV REPO_URL=https://github.com/AllenNeuralDynamics/aind-ophys-movie-qc.git
ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    h5py==3.9.0 \
    matplotlib==3.8.0 \
    numpy==1.26.0 \
    oasis-deconv==0.2.0 \
    scikit-image==0.21.0 \
    scikit-learn==1.3.1 \
    scipy==1.11.2 \
    aind-data-schema==1.1.1 \
    pydantic==2.8.2
