from setuptools import setup, find_packages

setup(
    name="carve",
    version="0.1.0",
    description="Carve: Coherent Morphological Loss Erosion for Fast Generalization",
    author="tasmaikeni13",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
    ],
)
