import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="bznames",
    version="0.0.1",
    author="Rafael Ladeira",
    description="Brazilian name generator using language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rladeira/bznames",
    packages=setuptools.find_packages(),
    install_requires=[],
    extras_require={"dev": []},
)
