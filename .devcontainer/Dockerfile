FROM nvcr.io/nvidia/tensorflow:24.12-tf2-py3

COPY requirements.txt .
RUN apt-get update \
    && python -m pip install --no-cache-dir -v -r requirements.txt