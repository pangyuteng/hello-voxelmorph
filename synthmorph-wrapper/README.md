

```

build_and_push.sh

docker run -it -u $(id -u):$(id -g) pangyuteng/synthmorph-wrapper:0.1.0 pip list | grep synth

docker run -it -u $(id -u):$(id -g) -w $PWD -v /cvibraid:/cvibraid pangyuteng/synthmorph-wrapper:0.1.0 bash

```
import setuptools

setuptools.setup(name='synthmorph_wrapper',
    version='0.1.0',
    description='synthmorph wrapper',
    url='https://github.com/pangyuteng/hello-voxelmorph/tree/main/synthmorph-wrapper',
    author='pangyuteng',
    license='MIT',
    py_modules=["synthmorph_wrapper"],
    install_requires=[
        'voxelmorph',
        'SimpleITK-SimpleElastix',
    ],
)

"""

ref https://setuptools.pypa.io/en/latest/userguide/quickstart.html


"""