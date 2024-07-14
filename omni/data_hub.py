"""Module for automated data collection."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from coord_generator import CoordinateGenerator
from google_api_handler import Handle

if TYPE_CHECKING:
    from PIL import Image
    from shapely.geometry import Point
from resources import Coordinate


class DataHub:
    """Automated data collection class."""

    def __init__(self: DataHub) -> None:
        """DataHub constructor."""
        self.google_api = Handle()
        generator = CoordinateGenerator()

        self.generator_dict = {}
        for country in generator.country_list:
            self.generator_dict[country] = generator.get_random_coord_generator(country)

    def save_img(self: DataHub, path: str, country: str, img: Image) -> None:
        """Save image in declared path.

        :param self: self
        :type self: DataHub
        :param path: place where country folder can be found
        :type path: str
        :param country: country name
        :type country: str
        :param img: image to save
        :type img: Image
        """
        country = country.lower().replace(" ", "_")
        country_dir = os.path.join(path,country)
        if not os.path.exists(country_dir):
            os.mkdir(country_dir)
        files = [f for f in os.listdir(country_dir) if os.path.isfile(os.path.join(country_dir, f))]
        filename = f"{len(files)}.png"
        img.save(country_dir+"/"+filename)

    def point_to_coord(self: DataHub, point: Point) -> Coordinate:
        """Convert shapely Point to Coordinate.

        :param self: self
        :type self: DataHub
        :param point: shapely point
        :type point: Point
        :return: converted coordinate
        :rtype: Coordinate
        """
        return Coordinate(point.x, point.y)

    def to_batches(self: DataHub, v: list, batch_size: int = 100) -> list[list]:
        """Split list into lists of max len of batch_size.

        :param self: self
        :type self: DataHub
        :param v: list to split
        :type v: list
        :param batch_size: max length of lists in return, defaults to 100
        :type batch_size: int, optional
        :return: list of splitted lists
        :rtype: list[list]
        """
        return [v[x:x+batch_size] for x in range(0, len(v), batch_size)]

    def get_stats(self: DataHub, path: str) -> dict:
        """Get dictionary of file count in specified path.

        :param self: self
        :type self: DataHub
        :param path: path to get stats from
        :type path: str
        :return: dict with dir name and file count
        :rtype: dict
        """
        dirs = os.listdir(path)
        stats_dict = {}
        for _dir in dirs:
            files = os.listdir(os.path.join(path, _dir))
            stats_dict[_dir] = len(files)
        return stats_dict

    def get_total_data_amount(self: DataHub, path: str) -> int:
        """Sum stats dictionary.

        :param self: self
        :type self: DataHub
        :param path: path to get stats from
        :type path: str
        :return: sum of files
        :rtype: int
        """
        return sum(self.get_stats(path).values())

    def get_data_standard_deviation(self: DataHub, path: str) -> float:
        """Calculate standard deviation of file counts.

        :param self: self
        :type self: DataHub
        :param path: path to get stats from
        :type path: str
        :return: standard deviation
        :rtype: float
        """
        return np.std(list(self.get_stats(path).values()))

    def get_data(
        self: DataHub,
        max_files: int,
        path: str,
        skip: list[str] | None = None,
        ) -> None:
        """Mainloop function.

        Collects data till each country has specified amount of files.

        :param self: self
        :type self: DataHub
        :param max_files: how much files each country can have
        :type max_files: int
        :param path: path to directory
        :type path: str
        :param skip: skip certain countries, defaults to None
        :type skip: list[str] | None, optional
        """
        print(f"Saving files till each country has {max_files}")
        if not skip:
            skip = []
        total = 0
        total_saved = 0
        stats = self.get_stats(path)
        for country, generator in self.generator_dict.items():
            if country in skip:
                continue
            saved = 0
            try:
                current_files = stats[country]
            except KeyError:
                current_files = 0
            batch_idx = 1
            while current_files < max_files:
                print(f"BATCH {batch_idx} for {country}, currently has {current_files} images")
                points = next(generator)
                total += 100
                coords = list(map(self.point_to_coord, points))
                images = self.google_api.get_full_panos(coords)
                for country_google, img in images:

                    if img:
                        self.save_img(path, country_google, img)
                        saved += 1
                        current_files += 1
                batch_idx += 1
            total_saved += saved
            print(f"{country} is done, has {current_files} images")
        print(f"Saved {total_saved} images")

if __name__ == "__main__":
    dh = DataHub()
    path = "data/countries/train"
    dh.get_data(300, path, ["israel"])
    print(dh.get_total_data_amount(path))
    print(dh.get_data_standard_deviation(path))
