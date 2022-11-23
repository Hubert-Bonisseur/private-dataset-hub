import logging

from setuptools import find_packages, setup

logger = logging.getLogger(__name__)

with open("README.md", "r") as f:
    README = f.read()

# upper limits are untested, not necessarily conflicting
# lower limits mostly to be python 3 compatible
REQUIRED_PACKAGES = ["datasets>=2.7.0", "gcsfs==2022.10.0"]

EXTRAS_REQUIRE = {
    "dev": [
        "pylint",
        "pytest",
        "pytest-cov",
    ]
}

setup(
    name="dataset-hub",
    version="0.DEV",
    description="Tools to upload and load datasets securely to gcloud while keeping the huggingface as showcase",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Hubert-Bonisseur/private-dataset-hub",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRAS_REQUIRE,
)
