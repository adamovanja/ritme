# ----------------------------------------------------------------------------
# Copyright (c) 2022, <developer name>.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from setuptools import find_packages, setup

import versioneer

setup(
    name="q2-plugin-name",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="BSD-3-Clause",
    packages=find_packages(),
    author="Michal Ziemski",
    author_email="ziemski.michal@gmail.com",
    description="This is a template for building a new QIIME 2 plugin.",
    url="https://github.com/bokulich-lab/q2-plugin-template",
    entry_points={
        "qiime2.plugins": ["q2-plugin-name=q2_plugin_name.plugin_setup:plugin"]
    },
    package_data={
        "q2_plugin_name": ["citations.bib"],
    },
    zip_safe=False,
)
