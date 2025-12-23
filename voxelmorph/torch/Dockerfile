FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
RUN apt-get update && apt-get install git vim -yq

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /tmp/.cache/matplotlib && chmod -R 777 /tmp/.cache/matplotlib
ENV MPLCONFIGDIR=/tmp/.cache/matplotlib
ENV TF_ENABLE_ONEDNN_OPTS=0

#COPY --chmod=777 torch/train.py /opt
#COPY --chmod=777 torch/register.py /opt
#COPY --chmod=777 torch/register_full_res.py /opt
