from shapely.geometry import Point, Polygon, MultiPolygon
from resources import all_coords
from extract_coords import Coordinator
from typing import List, Generator
import numpy as np
import random

class Coordinate_Generator:
    def __init__(self) -> None:
        parser = Coordinator()
        self.positive_dict, self.negative_dict = parser.get_multipolygon_dicts()
        self.country_list = list(self.positive_dict.keys())

    def is_in_positive(self, country: str, point: Point) -> bool:
        multipolygon_p = self.positive_dict[country]
        for polygon in multipolygon_p.geoms:
            if point.within(polygon):
                return True
    
        return False
    def is_in_negative(self, country: str, point: Point) -> bool:
        try:
            multipolygon_n = self.negative_dict[country]
        except KeyError:
            return False
        for poly_n in multipolygon_n.geoms:
            if point.within(poly_n):
                return True
        return False
        
    def get_random_coord_generator(self, country: str) -> Generator[List[Point], None, None]:

        multipolygon_p = self.positive_dict[country]

        areas = [poly.area for poly in multipolygon_p.geoms]
        weights = [float(i)/sum(areas) for i in areas]
        coord_list = []
        while True:

            poly = np.random.choice(multipolygon_p.geoms, p=weights)
            min_x, min_y, max_x, max_y = poly.bounds

            random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
            if self.is_in_positive(country, random_point) and (not self.is_in_negative(country, random_point)):
                coord_list.append(random_point)

            if len(coord_list) == 100:
                yield coord_list
                coord_list = []

if __name__ == "__main__":
    cg = Coordinate_Generator()
    generator = cg.get_random_coord_generator("uk")
    print(next(generator))