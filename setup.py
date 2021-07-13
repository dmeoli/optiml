import setuptools

setuptools.setup(name='optiml',
                 version='1.0',
                 author='Donato Meoli',
                 author_email='donato.meoli.95@gmail.com',
                 description='Optimizers for/and sklearn compatible Machine Learning models',
                 long_description=open('README.md').read(),
                 long_description_content_type='text/markdown',
                 license='MIT',
                 url='https://github.com/dmeoli/optiml',
                 packages=setuptools.find_packages(),
                 install_requires=open('requirements.txt').read().splitlines(),
                 python_requires='>=3.6')
