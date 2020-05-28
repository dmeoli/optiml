from setuptools import setup, find_packages

with open("README.md", "r") as file_stream:
    readme = file_stream.read()

with open("requirements.txt", "r") as file_stream:
    install_requires = file_stream.read().splitlines()

setup(name="optiml",
      version="0.0.2",
      author="Donato Meoli",
      author_email="donato.meoli.95@gmail.com",
      description="Optimizers for/and sklearn compatible Machine Learning models",
      long_description=readme,
      long_description_content_type="text/markdown",
      install_requires=install_requires,
      license="MIT",
      url="https://github.com/dmeoli/optiml",
      packages=find_packages(),
      python_requires='>=3.6')
