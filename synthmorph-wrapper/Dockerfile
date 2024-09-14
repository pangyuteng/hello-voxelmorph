FROM pangyuteng/voxelmorph

COPY --chmod=777 synthmorph_wrapper /opt/synthmorph_wrapper
COPY --chmod=777 pyproject.toml /opt
WORKDIR /opt
RUN pip install .
