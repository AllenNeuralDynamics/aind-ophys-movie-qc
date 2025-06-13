# hash:sha256:f7f2de5e8caaebf9319fc96309be2b4509aa68773ddf8f7fef31ac7eb84783a5
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.1.1 \
    aind-qcportal-schema==0.3.2 \
    h5py==3.9.0 \
    matplotlib==3.8.0 \
    numpy==1.26.0 \
    oasis-deconv==0.2.0 \
    pydantic==2.11.6 \
    scikit-image==0.21.0 \
    scikit-learn==1.3.1 \
    scipy==1.11.2
