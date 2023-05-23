"""2D template matching in PyTorch"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-2d-template-matching")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Josh Dickerson"
__email__ = "jdickerson@mrc-lmb.cam.ac.uk"
