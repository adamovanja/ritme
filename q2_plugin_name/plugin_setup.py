# ----------------------------------------------------------------------------
# Copyright (c) 2022, <developer name>.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import Citations, Plugin

from q2_plugin_name import __version__

citations = Citations.load("citations.bib", package="q2_plugin_name")

plugin = Plugin(
    name="plugin-name",
    version=__version__,
    website="https://github.com/bokulich-lab/q2-plugin-name",
    package="q2_plugin_name",
    description="This is a template for building a new QIIME 2 plugin.",
    short_description="",
)
