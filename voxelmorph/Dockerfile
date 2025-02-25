FROM tensorflow/tensorflow:2.11.0-gpu

RUN apt-get update && apt-get install git vim -yq

RUN /usr/bin/python3 -m pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /opt
RUN git clone https://github.com/voxelmorph/voxelmorph.git

COPY --chmod=777 utils.py /usr/local/lib/python3.8/dist-packages/voxelmorph/py/utils.py
COPY --chmod=777 train.py /opt/train.py