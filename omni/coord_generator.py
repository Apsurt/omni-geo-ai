from shapely.geometry import Point, Polygon, MultiPolygon
from resources import all_coords
import rancoord as rc

class Coordinate_Generator:
    def __init__(self) -> None:
        self._all_coords = []
        coordinates = all_coords
        self._polypolygon = MultiPolygon(map(Polygon, coordinates))
        self._total_area = sum(map(lambda x: x.area, self._polypolygon.geoms))

    def _get_random_coordinate(self, polygon:Polygon, num_locations:int=1) -> list[Point]:
        lats, lons, _ = rc.coordinates_randomizer(polygon = polygon, num_locations = num_locations)
        point_list = map(Point, lats, lons)
        return point_list

    def get_normalized_coord(self, total_num:int=1) -> list[Point]:
        coord_list = []
        for polygon in self._polypolygon.geoms:
            weight = int(polygon.area / self._total_area * total_num)
            coord_list += self._get_random_coordinate(polygon, weight)
        return coord_list
