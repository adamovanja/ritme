"""Set the top-level ``version`` field of a CITATION.cff file from a release tag.

Usage: python ci/update_citation.py CITATION.cff v1.4.5
"""

import argparse
import re
from typing import List, Optional

# matches ``tag_prefix`` in setup.cfg
TAG_PREFIX = "v"
VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+")
# only an unindented line, i.e. the top-level field
VERSION_LINE_PATTERN = re.compile(r"^version: .*$", flags=re.MULTILINE)


def version_from_tag(tag: str) -> str:
    version = tag[len(TAG_PREFIX) :] if tag.startswith(TAG_PREFIX) else tag
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"Tag {tag!r} is not of the form {TAG_PREFIX}X.Y.Z.")
    return version


def set_version(cff_text: str, version: str) -> str:
    updated, n_replaced = VERSION_LINE_PATTERN.subn(
        f"version: {version}", cff_text, count=1
    )
    if n_replaced == 0:
        raise ValueError("No top-level 'version' field found in the CITATION.cff.")
    return updated


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cff_path", help="Path to the CITATION.cff file")
    parser.add_argument("tag", help=f"Release tag, e.g. {TAG_PREFIX}1.4.5")
    args = parser.parse_args(argv)

    version = version_from_tag(args.tag)
    with open(args.cff_path, "r") as f:
        cff_text = f.read()
    with open(args.cff_path, "w") as f:
        f.write(set_version(cff_text, version))
    print(f"{args.cff_path}: version set to {version}")


if __name__ == "__main__":
    main()
