import argparse
from typing import Dict

import yaml
from jinja2 import Template


def load_meta_yaml(file_path: str) -> Dict:
    """
    Load and render a conda meta.yaml file with given default_versions.

    :param file_path: str, path to the meta.yaml file
    :return: dict, parsed meta.yaml data
    """
    with open(file_path, "r") as f:
        content = f.read()
        template = Template(content)
        rendered_content = template.render(load_setup_py_data=lambda: {})
        return yaml.safe_load(rendered_content)


def print_requirements(meta_yaml_data: Dict, req_type: str) -> None:
    """
    Print the 'run' requirements from the given meta.yaml data, either
    'conda' or 'pip' requirements, with a double equal sign between
    package name and version.

    :param meta_yaml_data: dict, parsed meta.yaml data
    """
    requirements = meta_yaml_data.get("requirements", {})
    if req_type == "conda":
        run_requirements = requirements.get("run", [])

        for requirement in run_requirements:
            if " " in requirement:
                package, version = requirement.split(" ")
                print(f"{package}=={version}")
            else:
                print(requirement)
    elif req_type == "pip":
        run_constrained = requirements.get("run_constrained", [])

        for constraint in run_constrained:
            if isinstance(constraint, dict) and "pip" in constraint:
                pip_packages = constraint["pip"]

                for pip_package in pip_packages:
                    print(pip_package)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract requirements from meta.yaml")
    parser.add_argument("meta_yaml_path", help="Path to meta.yaml file")
    parser.add_argument("req_type", help="Conda or pip requirements", default="conda")
    args = parser.parse_args()

    meta_yaml_data = load_meta_yaml(args.meta_yaml_path)
    print_requirements(meta_yaml_data, args.req_type)
