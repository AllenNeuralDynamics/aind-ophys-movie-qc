# hash:sha256:497d5b4244240d20e796fd64ae6ee596ef11f6fe5eb1ea53daec6c6b123bf8e0
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.1.1 \
    aind-qcportal-schema==0.3.2 \
    dask[distributed] \
    h5py==3.9.0 \
    matplotlib==3.8.0 \
    numpy==1.26.0 \
    oasis-deconv==0.2.0 \
    opencv-python \
    pydantic==2.11.6 \
    scanimage-tiff-reader \
    scikit-image==0.21.0 \
    scikit-learn==1.3.1 \
    scipy==1.11.2 \
    seaborn
