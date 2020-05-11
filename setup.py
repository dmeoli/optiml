from setuptools import setup, find_packages

setup(
    name="yase",
    version="0.0.1",
    author="Donato Meoli",
    author_email="donato.meoli.95@gmail.com",
    description="Yet Another Sklearn Extension",
    license="MIT",
    url="https://github.com/dmeoli/yase",
    include_package_data=True,
    zip_safe=False,
    packages=find_packages(),
    python_requires='>=3.6')
