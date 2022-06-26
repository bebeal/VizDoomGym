from setuptools import setup

setup(
    name="vizdoomgym",
    version="1.0",
    description="Highly Customizable Gym interface for ViZDoom.",
    author="Benjamin Noah Beal",
    author_email="bnoahbeal@gmail.com",
    install_requires=[
        "vizdoom",
        "gym==0.24.0",
        "numpy",
        "pygame",
        "opencv-python",
        "torch"
    ],
)
