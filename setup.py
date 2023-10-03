from setuptools import find_packages, setup

# Constant
HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str):
    """
    This is a function to get all the requirements.
    """
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [module.replace("\n", "") for module in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        print(requirements)


setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="insurance premium prediction",
    author="Rushikesh Shinde",
    license="MIT",
    author_email="rushikeshshindesandesh@gmail.com",
    install_requires=get_requirements("requirements.txt"),
)


# For testing perpose
# if __name__ == "__main__":
#     get_requirements("./requirements.txt")
