"""Helper module."""

from typing import NamedTuple

from PIL import Image


class Coordinate(NamedTuple):
    """Named tuple containing latitude and longitude."""

    lat: float
    lng: float

def combine_images(image: Image) -> Image:
    """Crops image.

    Image transformations, currently crops image
    as google returns 512x512 image and at zoom 0
    half of the image is black. Used to save
    storage space.

    :param image: Image to crop
    :type image: Image
    :return: Cropped Image
    :rtype: Image
    """
    return image.crop((0,0,512,256))
