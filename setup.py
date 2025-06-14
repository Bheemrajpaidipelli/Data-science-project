from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    hypen_e_dot = '-e .'
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip()]

        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)

    return requirements

setup(
    name="data_science_project",
    version='0.0.1',
    author='Bheemraj',
    author_email="paidipellibheemraj1444@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
