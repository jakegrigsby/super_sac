from setuptools import find_packages, setup

setup(
    name="uafbc",
    version="0.0.1",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    author="Jake Grigsby",
    author_email="jcg6dn@virginia.edu",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "gym",
        "numpy",
        "torch",
        "torchvision",
        "tqdm",
        "opencv-python",
        "Pillow",
        "scikit-image",
        "tensorboardX",
    ],
)
