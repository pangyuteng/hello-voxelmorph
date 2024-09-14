import setuptools

setuptools.setup(name='synthmorph_wrapper',
    version='0.1',
    description='synthmorph wrapper',
    url='https://github.com/pangyuteng/hello-voxelmorph/tree/main/wrapper',
    author='pangyuteng',
    license='MIT',
    py_modules=["synthmorph_wrapper"],
    install_requires=[
        'voxelmorph',
        'SimpleITK-SimpleElastix',
    ],
)

"""

ref https://docs.python.org/3.11/distutils/setupscript.html

"""