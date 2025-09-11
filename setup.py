from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Filter out comments and empty lines
requirements = [req for req in requirements if req and not req.startswith("#")]

setup(
    name="film-ai",
    version="1.0.0",
    description="Generate cinematic videos from text scripts using AI",
    author="Film AI Team",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'film-ai=main:main',
            'film-generate=scripts.generate:main',
            'film-train=scripts.train:main',
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)