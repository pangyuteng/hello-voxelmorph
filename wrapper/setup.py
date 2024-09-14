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
      include_package_data=True,
      zip_safe=False)

"""

ref https://gist.github.com/aveek22/4cd863ab1ce57d74dec5c4a1d4361bf6

"""