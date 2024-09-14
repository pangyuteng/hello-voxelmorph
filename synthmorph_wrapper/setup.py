import setuptools

setuptools.setup(name='synthmorph_wrapper',
    version='0.1',
    description='synthmorph wrapper',
    url='https://github.com/pangyuteng/hello-voxelmorph/tree/main/wrapper',
    author='pangyuteng',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
        'voxelmorph',
        'SimpleITK-SimpleElastix',
    ],
    py_modules=["synthmorph_wrapper"],
    package_dir={'':'synthmorph_wrapper'}, 
    include_package_data=True,
    zip_safe=False)

"""

ref https://docs.python.org/3.11/distutils/setupscript.html

"""