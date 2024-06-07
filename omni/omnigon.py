from shapely.geometry import Point, Polygon, MultiPolygon
from typing import Literal, List

class Omnigon:
    def __init__(self, name: str, poly_type: Literal["p", "n"], coord_list: List[Point]) -> None:
        self.name = name
        self.poly_type = poly_type
        self.polygon = Polygon(coord_list)

    @property
    def positive(self):
        return self.poly_type == "p"

    def __repr__(self):
        if self.positive:
            return f"{self.name}-positive"
        return f"{self.name}-negative"

    def __str__(self):
        return self.__repr__()
