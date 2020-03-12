import setuptools
from os import path
#TODO fix this by adding README.txt

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'),"r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="example-pkg-sbraun92", # Replace with your own username
    version="0.0.1",
    author="Soehnke Braun",
    author_email="soehnkebraun@posteo.net",
    description="Master thesis project "
                "- Reinforcement Learning for Stowage PLanning of RORO-vessels",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sbraun92/thesisSB",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        #"License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)