FROM nvcr.io/nvidia/tensorflow:24.12-tf2-py3

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y python3-pyqt5 \
    && python -m pip install --no-cache-dir -v -r requirements.txt