FROM tensorflow/tensorflow:2.11.0-gpu

RUN /usr/bin/python3 -m pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# WORKDIR /opt

