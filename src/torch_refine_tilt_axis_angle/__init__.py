"""Tilt-axis angle optimization for tilt series using common lines in PyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-refine-tilt-axis-angle")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Marten Chaillet"
__email__ = "martenchaillet@gmail.com"
