from setuptools import setup, find_packages

setup(
    name="iso_anti_sam",
    version="0.1.0",
    description="Isochoric and Coherent Anti-Sharpness-Aware Minimization",
    author="tasmaikeni13",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
    ],
)
