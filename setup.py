from setuptools import setup

with open("README.md", "r") as file_stream:
    readme = file_stream.read()

setup(name="optiml",
      version="0.0.7",
      author="Donato Meoli",
      author_email="donato.meoli.95@gmail.com",
      description="Optimizers for/and sklearn compatible Machine Learning models",
      long_description=readme,
      long_description_content_type="text/markdown",
      license="MIT",
      url="https://github.com/dmeoli/optiml",
      python_requires='>=3.6')
