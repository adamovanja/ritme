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
    name="q2-ritme",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="BSD-3-Clause",
    packages=find_packages(),
    author="Anja Adamov",
    author_email="anja.adamov@hest.ethz.ch",
    description="This is a QIIME 2 plugin for longitudinal modeling applied to "
    "microbial time-series.",
    url="https://github.com/adamovanja/q2-ritme",
    entry_points={"qiime2.plugins": ["q2-ritme=q2_ritme.plugin_setup:plugin"]},
    package_data={
        "q2_ritme": ["citations.bib"],
    },
    zip_safe=False,
)
