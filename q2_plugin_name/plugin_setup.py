# ----------------------------------------------------------------------------
# Copyright (c) 2022, <developer name>.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import Citations, Plugin

from q2_time import __version__

citations = Citations.load("citations.bib", package="q2_time")

plugin = Plugin(
    name="time",
    version=__version__,
    website="https://github.com/bokulich-lab/q2-time",
    package="q2_time",
    description="This is a template for building a new QIIME 2 plugin.",
    short_description="",
)
