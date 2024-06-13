"""KML Parser."""

from __future__ import annotations

import re
from pathlib import Path

from shapely.geometry import MultiPolygon, Point, Polygon


class Coordinator:
    """KML Parser."""

    def __init__(self: Coordinator) -> None:
        """Coordinator constructor."""
        with Path.open("omni/resources/omnigeocountries.kml") as f:
            self.lines = f.readlines()[4:]

    def get_polygon_dicts(self: Coordinator) -> list[Polygon]:
        """Parse KML file to dictionaries."""
        positive_dict = {}
        negative_dict = {}
        for idx, line in enumerate(self.lines):
            if "name" in line:
                full_name = re.findall(r"(?<=>)((.+)(?=<)|(?=<))", line.strip())[0][0]
                name = full_name[:-2]
                if "rkiye" in name:
                    #no comment...
                    name = "türkiye"
                if "union" in name:
                    #eh....
                    name = "réunion"
                poly_type = full_name[-1]
                if poly_type == "p":
                    current_dict = positive_dict
                if poly_type == "n":
                    current_dict = negative_dict
                if name not in current_dict:
                    current_dict[name] = []
            if "coordinates" in line and "</" not in line:
                coords = re.findall(r"([-\d]+\.\d+)", self.lines[idx+1].strip())
                coords = list(map(float, coords))
                lons = coords[::2]
                lats = coords[1::2]
                coords = list(zip(lats, lons))
                coords = list(map(self._tuple_to_point, coords))
                current_dict[name].append(coords)
        for name, coords in positive_dict.items():
            poly_list = []
            for coord in coords:
                new_polygon = Polygon(coord)
                poly_list.append(new_polygon)
            positive_dict[name] = poly_list
        for name, coords in negative_dict.items():
            poly_list = []
            for coord in coords:
                new_polygon = Polygon(coord)
                poly_list.append(new_polygon)
            negative_dict[name] = poly_list
        return positive_dict, negative_dict

    def _tuple_to_point(self: Coordinator, coord: tuple) -> Point:
        return Point(coord[0], coord[1])

    def get_multipolygon_dicts(self: Coordinator) -> MultiPolygon:
        """Convert Polygons to MultiPolygons and returns dictionary."""
        positive_dict, negative_dict = self.get_polygon_dicts()
        for name, polygons in positive_dict.items():
            positive_dict[name] = MultiPolygon(polygons)
        for name, polygons in negative_dict.items():
            negative_dict[name] = MultiPolygon(polygons)
        return positive_dict, negative_dict
