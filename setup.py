import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="learning_ifc",
  version="0.0.1",
  author="Michael Jendrusch",
  author_email="jendrusch@stud.uni-heidelberg.de",
  description="Machine learning for imaging flow cytometry.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/mjendrusch/learning-ifc/",
  packages=setuptools.find_packages(),
  classifiers=(
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ),
)
