from setuptools import find_packages, setup

__version__ = "0.0.0"
REPO_NAME = "dog-classification"
AUTHOR_USER_NAME = "RoyalNam"
AUTHOR_EMAIL = "hoangnam242003@gmail.com"
SRC_REPO = "cnnClassifier"

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    package_dir={"": "src"},
    packages=find_packages(where="src")
)
