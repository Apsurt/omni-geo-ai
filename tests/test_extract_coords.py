
from shapely import Point, MultiPolygon, Polygon
from omni import Coordinator

coordinator = Coordinator()
with open("omni/resources/countries.txt") as f:
    country_list = f.readlines()
country_list = list(map(lambda x: x.strip().lower().replace(" ", "_"), country_list))

def test_tuple_to_point():
    coords = (1,1)
    coords = coordinator._tuple_to_point(coords)
    assert coords == Point(1,1)

def test_get_multipolygon_dicts():
    positive_dict, negative_dict = coordinator.get_multipolygon_dicts()
    for key, value in positive_dict.items():
        assert key in country_list
        assert isinstance(value, MultiPolygon)
        assert len(value.geoms) > 0
        for poly in value.geoms:
            assert isinstance(poly, Polygon)
