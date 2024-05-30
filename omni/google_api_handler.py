"""
Docstring
"""

import os
import io
import cProfile
from typing import List, NamedTuple, Dict
import json
import numpy as np
import requests
import PIL
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

class Coordinate(NamedTuple):
    lat: float
    lng: float

class Handle:
    def __init__(self) -> None:
        self.__session_id = os.getenv("GOOGLE_SESSION")
        self.__key = os.getenv("GOOGLE_KEY")

    def get_session_id(self) -> str:
        headers = {"Content-Type": "application/json"}
        params = {"mapType":  "streetview",
                  "language": "en-US",
                  "region":   "US"}
        url = f'https://tile.googleapis.com/v1/createSession?key={self.__key}'
        response = requests.post(url, headers=headers, params=params, timeout=10)
        return json.loads(response.text)["session"]
    
    def update_session_id(self) -> None:
        new_session_id = self.get_session_id()
        self.__session_id = new_session_id
        os.environ["GOOGLE_SESSION"] = new_session_id

    def get_pano_ids(self, coordinates: Coordinate | List[Coordinate], radius: int = 50) -> List[str]:
        if isinstance(coordinates, Coordinate):
            coordinates = [coordinates]
        if len(coordinates) > 100:
            raise RuntimeError("Too much coordinates for one response.")
        headers = {"Content-Type": "application/json"}
        payload = {"locations": [], "radius": radius}
        url = f"https://tile.googleapis.com/v1/streetview/panoIds?session={self.__session_id}&key={self.__key}"
        for coord in coordinates:
            loc = {"lat": coord.lat, "lng": coord.lng}
            payload["locations"].append(loc)
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        return json.loads(response.text)["panoIds"]
    
    def get_metadata(self, pano_id) -> Dict:
        headers = {"Content-Type": "application/json"}
        params = {"mapType":  "streetview",
                  "language": "en-US",
                  "region":   "US"}
        url = f"https://tile.googleapis.com/v1/streetview/metadata?session={self.__session_id}&key={self.__key}&panoId={pano_id}"
        response = requests.get(url, headers=headers, params=params, timeout=10)
        return json.loads(response.text)

    def get_tile(self, pano_id: str, z: int, x: int, y: int) -> Image:
        headers = {"Content-Type": "application/json"}
        url = f"https://tile.googleapis.com/v1/streetview/tiles/{z}/{x}/{y}?session={self.__session_id}&key={self.__key}&panoId={pano_id}"
        image_bytes = requests.get(url, headers=headers, timeout=10)
        try:
            return Image.open(io.BytesIO(image_bytes.content))
        except PIL.UnidentifiedImageError as e:
            print(image_bytes.text)
            raise e
    
    def combine_images(self, images: List[List[Image]], width: int, height: int) -> Image:
        image_width, image_height = images[0][0].size
        total_width = image_width * width
        total_height = image_height * height
        final_image = Image.new("RGB", (total_width, total_height))
        
        for y, y_offset in enumerate(range(0, total_height, image_height)):
            for x, x_offset in enumerate(range(0, total_width, image_width)):
                final_image.paste(images[y][x], (x_offset,y_offset))
        return final_image

    def get_full_panos(self, coordinates: Coordinate | List[Coordinate], z: int) -> List[Image]:
        pano_ids = self.get_pano_ids(coordinates)
        combined_images = []
        for pano_id in pano_ids:
            metadata = self.get_metadata(pano_id)
            #image_height = metadata[I]
            max_x = 1
            max_y = 1
            print(max_x,max_y)
            images = []
            for y in range(max_y):
                images.append([])
                for x in range(max_x):
                    print(x, y)
                    image = self.get_tile(pano_id, z, x, y)
                    images[-1].append(image)
            combined_image = self.combine_images(images, max_x, max_y)
            combined_image.show()
            combined_images.append(combined_image)
        return combined_images

def main():
    handle = Handle()
    polito = Coordinate(45.0623522053154, 7.66269291001806)
    wro = Coordinate(51.10926886416323, 17.03223364904242)
    coords = [polito, wro]
    handle.get_full_panos(coords, 0)

if __name__ == "__main__":
    main()
    #cProfile.run("main()", sort="cumtime")
