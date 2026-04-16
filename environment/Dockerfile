# hash:sha256:94f4145f68b6fc6ce2e7e87573137282e79fb09baf0bd0875b1e4704f14c8c60
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:v5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

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
