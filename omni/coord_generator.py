from shapely.geometry import Point, Polygon, MultiPolygon
from resources import utils
import rancoord as rc

class Coordinate_Generator:
    def __init__(self) -> None:
        self._all_coords = []
        coordinates = utils.all_coords
        self._polypolygon = MultiPolygon(map(Polygon, coordinates))
        self._total_area = sum(map(lambda x: x.area, self._polypolygon.geoms))

    def get_random_coordinate(self, polygon:Polygon, num_locations:int=1) -> list[Point]:
        lats, lons, _ = rc.coordinates_randomizer(polygon = polygon, num_locations = num_locations)
        pointlist = map(Point, lats, lons)
        return pointlist

    def get_normalized_coord(self, total_num:int=1) -> list[Point]:
        tempcoords = []
        for polygon in self._polypolygon.geoms:
            weight = int(polygon.area / self._total_area * total_num)
            tempcoords += self.get_random_coordinate(polygon, weight)
        return tempcoords
