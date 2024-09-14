from setuptools import setup

setup(name='synthmorph_wrapper',
      version='0.1',
      description='synthmorph wrapper',
      url='https://github.com/sean-mcclure/datapeek_py',
      author='Sean McClure',
      author_email='sean.mcclure@example.com',
      license='MIT',
      packages=['synthmorph_wrapper'],
      install_requires=[
          'voxelmorph',
          'SimpleITK-SimpleElastix',
      ],
      include_package_data=True,
      zip_safe=False)
