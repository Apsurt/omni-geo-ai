import re
from typing import List, Tuple
from shapely.geometry import Point, Polygon, MultiPolygon
from omnigon import Omnigon

class Coordinator:
    def __init__(self) -> None:
        with open("omni/resources/omnigeocountries.kml", "r") as f:
            self.lines = f.readlines()[4:]

    def _get_all_polygons(self) -> List[Polygon]:
        _polygons_all = []
        positive_dict = {}
        negative_dict = {}
        for idx, line in enumerate(self.lines):
            if "name" in line:
                full_name = re.findall(r"(?<=>)((.+)(?=<)|(?=<))", line.strip())[0][0]
                name = full_name[:-2]
                poly_type = full_name[-1]
                if poly_type == "p":
                    current_dict = positive_dict
                if poly_type == "n":
                    current_dict = negative_dict
                if not name in current_dict:
                    current_dict[name] = []
            if "coordinates" in line and not "</" in line:
                coords = re.findall(r"([-\d]+\.\d+)", self.lines[idx+1].strip())
                coords = list(map(float, coords))
                lons = coords[::2]
                lats = coords[1::2]
                coords = list(zip(lats, lons))
                coords = list(map(self._tuple_to_point, coords))
                current_dict[name].append(coords)
        for name, coords in positive_dict.items():
            for coord in coords:
                new_omnigon = Omnigon(name, "p", coord)
                _polygons_all.append(new_omnigon)
        for name, coords in negative_dict.items():
            for coord in coords:
                new_omnigon = Omnigon(name, "n", coord)
                _polygons_all.append(new_omnigon)
        return _polygons_all

    def _tuple_to_point(self, coord: Tuple) -> Point:
        return Point(coord[0], coord[1])

    def get_polypoly(self) -> MultiPolygon:
        return MultiPolygon(self._get_all_polygons())

if __name__ == "__main__":
    coord = Coordinator()
    coord.get_polypoly()
