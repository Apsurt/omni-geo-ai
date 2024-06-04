from typing import List
import os
from PIL import Image
from shapely.geometry import Point
from google_api_handler import Handle
from coord_generator import Coordinate_Generator
from resources import Coordinate

class DataHub:
    def __init__(self) -> None:
        self.google_api = Handle()
        self.generator = Coordinate_Generator()

    def save_img(self, path: str, country: str, img: Image) -> None:
        country = country.lower().replace(" ", "_")
        country_dir = os.path.join(path,country)
        if not os.path.exists(country_dir):
            os.mkdir(country_dir)
        files = [f for f in os.listdir(country_dir) if os.path.isfile(os.path.join(country_dir, f))]
        filename = f"{len(files)}.png"
        img.save(country_dir+"/"+filename)

    def point_to_coord(self, point: Point) -> Coordinate:
        coord = Coordinate(point.x, point.y)
        return coord

    def to_batches(self, v: List, batch_size: int = 100) -> List[List]:
        return [v[x:x+batch_size] for x in range(0, len(v), batch_size)]

    def save_data(self, n: int, path: str) -> None:
        saved = 0
        points = self.generator.get_normalized_coord(n)
        coords = list(map(self.point_to_coord, points))
        batches = self.to_batches(coords)
        for idx, batch in enumerate(batches):
            print(f"BATCH {idx+1}")
            images = self.google_api.get_full_panos(batch, 0)
            for country, img in images:
                if img:
                    self.save_img(path, country, img)
                    saved += 1
        print(f"Saved {saved} images")
        print(f"Hit percentage: {saved/n*100}%")

if __name__ == "__main__":
    dh = DataHub()
    dh.save_data(10000, "data/countries/train")
