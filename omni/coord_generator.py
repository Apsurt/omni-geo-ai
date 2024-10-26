"""Module containing Random Coordinate Generator class."""

from __future__ import annotations

from typing import Generator

import numpy as np
from extract_coords import Coordinator
from shapely.geometry import Point


class CoordinateGenerator:
    """Class for generating random coordinates."""

    def __init__(self: CoordinateGenerator) -> None:
        """Class constructor."""
        parser = Coordinator()
        self.positive_dict, self.negative_dict = parser.get_multipolygon_dicts()
        self.country_list = list(self.positive_dict.keys())

    def is_in_positive(self: CoordinateGenerator, country: str, point: Point) -> bool:
        """Check if coordinate is in positive polygon.

        :param self: self
        :type self: CoordinateGenerator
        :param country: Country name
        :type country: str
        :param point: Point to check
        :type point: Point
        :return: is point in positive polygon
        :rtype: bool
        """
        multipolygon_p = self.positive_dict[country]
        return any(point.within(polygon) for polygon in multipolygon_p.geoms)

    def is_in_negative(self: CoordinateGenerator, country: str, point: Point) -> bool:
        """Check if coordinate is in negative polygon.

        :param self: self
        :type self: CoordinateGenerator
        :param country: Country name
        :type country: str
        :param point: Point to check
        :type point: Point
        :return: is point in negative polygon
        :rtype: bool
        """
        try:
            multipolygon_n = self.negative_dict[country]
        except KeyError:
            return False
        return any(point.within(poly_n) for poly_n in multipolygon_n.geoms)

    def get_random_coord_generator(
        self: CoordinateGenerator,
        country: str,
        ) -> Generator[list[Point], None, None]:
        """Create generator yielding list of points.

        :param self: self
        :type self: CoordinateGenerator
        :param country: country to make the generator forc
        :type country: str
        :rtype: Generator[list[Point], None, None]
        """
        max_points = 100
        try:
            multipolygon_p = self.positive_dict[country]
        except KeyError as e:
            print(self.positive_dict.keys())
            raise e

        areas = [poly.area for poly in multipolygon_p.geoms]
        weights = [float(i)/sum(areas) for i in areas]
        coord_list = []
        while True:

            poly = np.random.choice(multipolygon_p.geoms, p=weights)
            min_x, min_y, max_x, max_y = poly.bounds

            lat = np.random.uniform(min_x, max_x)
            lon = np.random.uniform(min_y, max_y)
            random_point = Point([lat, lon])
            if self.is_in_positive(country, random_point) and \
               (not self.is_in_negative(country, random_point)):
                coord_list.append(random_point)

            if len(coord_list) == max_points:
                yield coord_list
                coord_list = []
