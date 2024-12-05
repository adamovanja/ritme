# ----------------------------------------------------------------------------
# Copyright (c) 2023, Anja Adamov.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from setuptools import find_packages, setup

import versioneer

setup(
    name="ritme",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="BSD-3-Clause",
    packages=find_packages(),
    author="Anja Adamov",
    author_email="anja.adamov@hest.ethz.ch",
    description="Target-driven optimization of feature representation and model "
    "selection for next-generation sequencing data",
    url="https://github.com/adamovanja/ritme",
    package_data={
        "ritme": ["citations.bib"],
        "ritme.tests": ["data/*"],
    },
    entry_points={
        "console_scripts": [
            "ritme=ritme.cli:app",
        ],
    },
    zip_safe=False,
)
