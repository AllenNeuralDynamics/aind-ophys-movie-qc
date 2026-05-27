# hash:sha256:75839a2bb31c86a64adf1958e8b2eb7157252f5a10e846b139bc37aa7486787f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.2.0 \
    aind-qcportal-schema==0.3.2 \
    h5py==3.9.0 \
    matplotlib==3.8.0 \
    numpy==1.26.0 \
    oasis-deconv==0.2.0 \
    pydantic==2.11.6 \
    scikit-image==0.21.0 \
    scikit-learn==1.3.1 \
    scipy==1.11.2
