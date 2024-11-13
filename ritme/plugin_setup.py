# ----------------------------------------------------------------------------
# Copyright (c) 2023, Anja Adamov.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import Citations, Plugin

from ritme import __version__

citations = Citations.load("citations.bib", package="ritme")

plugin = Plugin(
    name="ritme",
    version=__version__,
    website="https://github.com/adamovanja/ritme",
    package="ritme",
    description="Target-driven optimization of feature representation and model "
    "selection for next-generation sequencing data",
    short_description="",
)
