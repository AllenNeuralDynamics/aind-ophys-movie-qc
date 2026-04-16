# hash:sha256:04b629546595c1ad87d8d67af73eda516d3db166d4945007310fa499b3c07f15
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:v5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        g++=4:11.2.0-1ubuntu1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==2.6.0 \
    aind-qcportal-schema==0.6.4 \
    h5py==3.16.0 \
    matplotlib==3.10.8 \
    numpy==2.4.4 \
    oasis-deconv==0.2.1 \
    pydantic==2.11.10 \
    pydantic-settings \
    pytz \
    scikit-image==0.26.0 \
    scikit-learn==1.8.0 \
    scipy==1.17.1
