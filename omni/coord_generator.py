from shapely.geometry import Point, Polygon, MultiPolygon
from resources import all_coords
from extract_coords import Coordinator
import rancoord as rc
import random

class Coordinate_Generator:
    def __init__(self) -> None:
        self._all_coords = []
        coordinates = all_coords
        self.coordinator = Coordinator()
        self._polypolygon = self.coordinator.get_polypoly()
        self._total_area = sum(map(lambda x: x.area, self._polypolygon.geoms))

    def _get_random_coordinate(self, polygon:Polygon, num_locations:int=1) -> list[Point]:
        # lats, lons, _ = rc.coordinates_randomizer(polygon = polygon, num_locations = num_locations)
        lats, lons = self._draw_points(polygon, num_locations)
        point_list = map(Point, lats, lons)
        return point_list

    def _draw_points(self, polygon:Polygon, num_locations:int) -> list[Point]:
        min_x, min_y, max_x, max_y = polygon.bounds
        points = []
        while len(points) < num_locations:
            random_point = Point(
                [random.uniform(min_x, max_x), random.uniform(min_y, max_y)]
            )
            if self._is_correct_point(random_point):
                points.append(random_point)
        lat = [point.x for point in points]
        lon = [point.y for point in points]
        return lat, lon

    def _is_correct_point(self, point: Point) -> bool:
        count = 0
        for polygon in self._polypolygon.geoms:
            if point.within(polygon):
                count += 1
        if count >= 2:
            print(count)
        return False if count % 2 == 0 else True
        # if count % 2 == 0:
        #     return False
        # else:
        #     return True
        
    def get_normalized_coord(self, total_num:int=1) -> list[Point]:
        coord_list = []
        for polygon in self._polypolygon.geoms:
            weight = int(polygon.area / self._total_area * total_num)
            coord_list += self._get_random_coordinate(polygon, weight)
        return coord_list

def main():
    c = Coordinate_Generator()
    a = c.get_normalized_coord(1000)


main()