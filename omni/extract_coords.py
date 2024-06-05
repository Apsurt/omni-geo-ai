from bs4 import BeautifulSoup
from shapely.geometry import Point, Polygon, MultiPolygon

class Coordinator:
    def __init__(self) -> None:
        self._soup = BeautifulSoup(open("./omni/resources/omnigeo.kml",
                                         encoding="UTF-8"), 'html.parser')

    def _get_all_polygons(self) -> list[Polygon]:
        _polygons_all = []
        for coordinate_chunk in self._soup.find_all("coordinates"):
            
            _polygons_all.append(self._points_to_polygon(coordinate_chunk))
        return _polygons_all

    def _points_to_polygon(self, coord_chunk:str) -> Polygon:
        temp_coordinates = (str(coord_chunk)[14:-14]).split(",0 ")
        del temp_coordinates[-1]
        poly_points = map(lambda x: Point(list(map(float, x.split(",")))), temp_coordinates)
        return Polygon(poly_points)

    def get_polypoly(self) -> MultiPolygon:
        return MultiPolygon(self._get_all_polygons())