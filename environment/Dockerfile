# hash:sha256:75839a2bb31c86a64adf1958e8b2eb7157252f5a10e846b139bc37aa7486787f
FROM python:3.9-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

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


RUN mkdir -p /data /results /scratch /raw /zstacks
COPY code/ /capsule/code
RUN chmod +x /capsule/code/run

