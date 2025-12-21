#docker build -t pangyuteng/voxelmorph:0.1.2-tf -f Dockerfile .
#docker push pangyuteng/voxelmorph:0.1.2-tf

docker build -t pangyuteng/voxelmorph:0.1.2-tf -f Dockerfile.keras224 .
docker push pangyuteng/voxelmorph:0.1.2-tf