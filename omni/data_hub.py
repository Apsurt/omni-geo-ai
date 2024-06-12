from typing import List, Dict
import os
import numpy as np
from PIL import Image
from shapely.geometry import Point
from google_api_handler import Handle
from coord_generator import Coordinate_Generator
from resources import Coordinate

class DataHub:
    def __init__(self) -> None:
        self.google_api = Handle()
        generator = Coordinate_Generator()

        self.generator_dict = {}
        for country in generator.country_list:
            self.generator_dict[country] = generator.get_random_coord_generator(country)

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
    
    def get_stats(self, path: str) -> Dict:
        dirs = os.listdir(path)
        stats_dict = {}
        for _dir in dirs:
            files = os.listdir(os.path.join(path, _dir))
            stats_dict[_dir] = len(files)
        return stats_dict
    
    def get_total_data_amount(self, path: str):
        return sum(self.get_stats(path).values())
    
    def get_data_standard_deviation(self, path: str):
        return np.std(list(self.get_stats(path).values()))

    def get_data(self, max_files: int, path: str, skip: List[str] = []) -> None:
        print(f"Saving files till each country has {max_files}")
        total = 0
        total_saved = 0
        stats = self.get_stats(path)
        for country, generator in self.generator_dict.items():
            if country in skip:
                continue
            saved = 0
            try:
                current_files = stats[country]
            except KeyError:
                current_files = 0
            batch_idx = 1
            while current_files < max_files:
                print(f"BATCH {batch_idx} for {country}, currently has {current_files} images")
                points = next(generator)
                total += 100
                coords = list(map(self.point_to_coord, points))
                images = self.google_api.get_full_panos(coords)
                for country_google, img in images:

                    if img:
                        self.save_img(path, country_google, img)
                        saved += 1
                        current_files += 1
                batch_idx += 1
            total_saved += saved
            print(f"{country} is done, has {current_files} images")
        print(f"Saved {total_saved} images")

if __name__ == "__main__":
    dh = DataHub()
    path = "data/countries/validate"
    dh.get_data(10, path)
    print(dh.get_total_data_amount(path))
    print(dh.get_data_standard_deviation(path))
