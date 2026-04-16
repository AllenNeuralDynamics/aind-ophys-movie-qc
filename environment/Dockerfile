# hash:sha256:8371fe6b209174366ee47dee9df26a151a9d3f28bff66314acd86e6b3d475633
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:v5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    aind-data-schema \
    aind-qcportal-schema \
    h5py \
    matplotlib \
    numpy \
    oasis-deconv \
    pydantic \
    scikit-image \
    scikit-learn \
    scipy
