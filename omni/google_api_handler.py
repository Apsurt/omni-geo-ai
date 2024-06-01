"""
Docstring
"""

import os
import io
import cProfile
from typing import List, NamedTuple, Dict
import json
import re
import requests
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
    
    def analyze_response(self, response) -> None:
        code = int(re.findall(r"\d+", str(response))[0])
        match code:
            case 200:
                pass
            case 400:
                raise RuntimeError(f"Bad request\n{response.text}")

    def get_session_id(self) -> str:
        headers = {"Content-Type": "application/json"}
        params = {"mapType":  "streetview",
                  "language": "en-US",
                  "region":   "US"}
        url = f'https://tile.googleapis.com/v1/createSession?key={self.__key}'
        response = requests.post(url, headers=headers, params=params, timeout=10)
        self.analyze_response(response)
        return json.loads(response.text)["session"]

    def update_session_id(self) -> None:
        new_session_id = self.get_session_id()
        self.__session_id = new_session_id
        os.environ["GOOGLE_SESSION"] = new_session_id

    def get_pano_ids(self, coordinates: Coordinate | List[Coordinate]) -> List[str]:
        if isinstance(coordinates, Coordinate):
            coordinates = [coordinates]
        if len(coordinates) > 100:
            raise RuntimeError("Too much coordinates for one response.")
        pano_ids = []
        headers = {"Content-Type": "application/json"}
        for coord in coordinates:
            url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={coord.lat},{coord.lng}&key={self.__key}"
            response = requests.post(url, headers=headers, timeout=10)
            response_json = json.loads(response.text)
            if response_json["status"] == "OK":
                pano_ids.append(response_json["pano_id"])
            else:
                pano_ids.append(None)
        return pano_ids

    def get_metadata(self, pano_id) -> Dict:
        headers = {"Content-Type": "application/json"}
        params = {"mapType":  "streetview",
                  "language": "en-US",
                  "region":   "US"}
        url = f"https://tile.googleapis.com/v1/streetview/metadata?session={self.__session_id}&key={self.__key}&panoId={pano_id}"
        response = requests.get(url, headers=headers, params=params, timeout=10)
        self.analyze_response(response)
        return json.loads(response.text)

    def get_tile(self, pano_id: str, z: int, x: int, y: int) -> Image:
        headers = {"Content-Type": "application/json"}
        url = f"https://tile.googleapis.com/v1/streetview/tiles/{z}/{x}/{y}?session={self.__session_id}&key={self.__key}&panoId={pano_id}"
        image_bytes = requests.get(url, headers=headers, timeout=10)
        self.analyze_response(image_bytes)
        return Image.open(io.BytesIO(image_bytes.content))

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
            images = []
            max_x = 0
            max_y = 0
            y=0
            run = True
            while run:
                images.append([])
                x = 0
                while run:
                    try:
                        image = self.get_tile(pano_id, z, x, y)
                    except RuntimeError:
                        if x == 0:
                            run = False
                        break
                    images[-1].append(image)
                    if run:
                        x += 1
                        max_x = max(x,max_x)
                if run:
                    y += 1
                    max_y = max(y,max_y)
            combined_image = self.combine_images(images, max_x, max_y)
            combined_image.show()
            combined_images.append(combined_image)
        return combined_images

def main():
    handle = Handle()
    polito = Coordinate(45.0623522053154, 7.66269291001806)
    wro = Coordinate(51.10926886416323, 17.03223364904242)
    coords = [polito, wro]
    handle.get_full_panos(coords, 2)

if __name__ == "__main__":
    #cProfile.run("main()", sort="cumtime")
    main()
