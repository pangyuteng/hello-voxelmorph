
FROM tensorflow/tensorflow:2.7.0-gpu

#RUN apt-get update && apt-get install git vim -yq
#RUN /usr/bin/python3 -m pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt


WORKDIR /opt
RUN git clone https://github.com/adalca/neurite
WORKDIR /opt/neurite
RUN cd /opt/neurite && git checkout c7bb05d5dae47d2a79e0fe5a8284f30b2304d335 \
&& pip install .
WORKDIR /opt
RUN git clone https://github.com/voxelmorph/voxelmorph
RUN cd /opt/voxelmorph && git checkout 52dd120f3ae9b0ab0fde5d0efe50627a4528bc9f \
&& python3 setup.py install

RUN mkdir -p /tmp/.cache/matplotlib && chmod -R 777 /tmp/.cache/matplotlib
ENV MPLCONFIGDIR=/tmp/.cache/matplotlib
ENV TF_ENABLE_ONEDNN_OPTS=0

# COPY --chmod=777 train_128.py /opt
# COPY --chmod=777 train_256.py /opt
# COPY --chmod=777 register.py /opt
# COPY --chmod=777 register_full_res.py /opt
