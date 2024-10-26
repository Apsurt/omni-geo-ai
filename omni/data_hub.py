"""to do."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import googlemaps
from coord_generator import CoordinateGenerator
from dotenv import load_dotenv
from shapely.geometry import Point

if TYPE_CHECKING:
    import shapely

load_dotenv()

def get_hemisphere(point: shapely.geometry.point.Point) -> bool:
    """Check on which hemisphere the point is.

    :param point: Point to check
    :type point: shapely.geometry.point.Point
    :return: True for northern, False for southern.
    :rtype: bool
    """
    return point.x > 0

def get_elevation(points: list[shapely.geometry.point.Point]) -> list[float]:
    if len(points) > 512:
        msg = f"This endpoint can take maximum of 512 coordinates.{len(points)} is too much."
        raise ValueError(msg)
    client = googlemaps.Client(key=os.getenv("GOOGLE_KEY"), timeout=5)
    points = points.copy()
    for i in range(len(points)):
        points[i] = (points[i].x, points[i].y)
    response = client.elevation(points)
    return [data["elevation"] for data in response]

cg = CoordinateGenerator()
generator = cg.get_random_coord_generator("poland")

points = next(generator)
print(points)
print(get_hemisphere(points[0]))
print(get_elevation(points))
