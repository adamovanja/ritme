# ----------------------------------------------------------------------------
# Copyright (c) 2023, Anja Adamov.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import Citations, Plugin

from q2_ritme import __version__

citations = Citations.load("citations.bib", package="q2_ritme")

plugin = Plugin(
    name="ritme",
    version=__version__,
    website="https://github.com/adamovanja/q2-ritme",
    package="q2_ritme",
    description="This is a QIIME 2 plugin for longitudinal modeling applied to "
    "microbial time-series.",
    short_description="",
)
