FROM pangyuteng/voxelmorph

COPY synthmorph_wrapper /opt/synthmorph_wrapper
COPY setup.py /opt
WORKDIR /opt
RUN python setup.py install
