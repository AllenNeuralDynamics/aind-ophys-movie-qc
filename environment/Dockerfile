# hash:sha256:a8a30a1db6bdac859dffda754efc3670040a5b96d65a03eb6bd65d6cd695cf6e
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

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
    scipy==1.11.2
