
FROM tensorflow/tensorflow:2.7.0-gpu-jupyter

RUN mkdir -p /tmp/.cache/matplotlib && chmod -R 777 /tmp/.cache/matplotlib
ENV MPLCONFIGDIR=/tmp/.cache/matplotlib
ENV TF_ENABLE_ONEDNN_OPTS=0

COPY pystrum /opt/pystrum
WORKDIR /opt/pystrum
RUN pip install .

COPY neurite /opt/neurite
WORKDIR /opt/neurite
RUN pip install .

COPY voxelmorph /opt/voxelmorph
WORKDIR /opt/voxelmorph
RUN pip install .

#COPY requirements.txt /tmp/requirements.txt
#RUN pip3 install -r /tmp/requirements.txt


# COPY --chmod=777 train_128.py /opt
# COPY --chmod=777 train_256.py /opt
# COPY --chmod=777 register.py /opt
# COPY --chmod=777 register_full_res.py /opt
