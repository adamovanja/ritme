import argparse
from typing import Dict

import yaml
from jinja2 import Template


def load_meta_yaml(file_path: str, default_versions: Dict[str, str]) -> Dict:
    """
    Load and render a conda meta.yaml file with given default_versions.

    :param file_path: str, path to the meta.yaml file
    :param default_versions: dict, default version values for Jinja2 variables
    :return: dict, parsed meta.yaml data
    """
    with open(file_path, "r") as f:
        content = f.read()
        template = Template(content)
        rendered_content = template.render(
            load_setup_py_data=lambda: {}, **default_versions
        )
        return yaml.safe_load(rendered_content)


def print_requirements(meta_yaml_data: Dict) -> None:
    """
    Print the 'run' requirements from the given meta.yaml
    data with a double equal sign between package name and version.

    :param meta_yaml_data: dict, parsed meta.yaml data
    """
    requirements = meta_yaml_data.get("requirements", {})
    run_requirements = requirements.get("run", [])

    for requirement in run_requirements:
        if " " in requirement:
            package, version = requirement.split(" ")
            print(f"{package}=={version}")
        else:
            print(requirement)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract requirements from meta.yaml")
    parser.add_argument("meta_yaml_path", help="Path to meta.yaml file")
    parser.add_argument(
        "--qiime2_epoch", help="Set version for qiime2_epoch", default="2023.2"
    )
    parser.add_argument("--python", help="Default Python version", default="3.8")
    args = parser.parse_args()

    default_versions = {
        "qiime2_epoch": args.qiime2_epoch,
        "python": args.python,
    }

    meta_yaml_data = load_meta_yaml(args.meta_yaml_path, default_versions)
    print_requirements(meta_yaml_data)