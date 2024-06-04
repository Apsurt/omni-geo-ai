"""
Docstring
"""

import os
import io
import cProfile
from typing import List, NamedTuple, Dict, Tuple
import json
import re
import requests
from PIL import Image
from resources import Coordinate
from dotenv import load_dotenv
load_dotenv()

class Handle:
    def __init__(self) -> None:
        self.__session_id = os.getenv("GOOGLE_SESSION")
        self.__key = os.getenv("GOOGLE_KEY")
        self.update_session_id()

    def analyze_response(self, response) -> None:
        code = int(re.findall(r"\d+", str(response))[0])
        match code:
            case 200:
                pass
            case 400:
                raise RuntimeError(f"Bad request\n{response.text}")
            case 404:
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
    
    def get_country_from_metadata(self, metadata: Dict) -> Dict:
        components = metadata["addressComponents"]
        for component in components:
            if "country" in component["types"]:
                return component

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
        final_image = final_image.crop((0,0,512,256))
        return final_image

    def get_full_panos(self, coordinates: Coordinate | List[Coordinate], z: int) -> List[Tuple[str, Image]]:
        pano_ids = self.get_pano_ids(coordinates)
        n = len(pano_ids)
        combined_images = []
        for idx, pano_id in enumerate(pano_ids):
            if pano_id is None:
                combined_images.append((None, None))
                continue
            metadata = self.get_metadata(pano_id)
            country_dict = self.get_country_from_metadata(metadata)
            country = country_dict["longName"]
            print(f"{idx+1}/{n}")
            print(f"Getting {pano_id}, {country}")
            print()
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
            combined_images.append((country, combined_image))
        return combined_images

def save_img(country, img):
    main_dir = "data/temp/"
    country = country.lower().replace(" ", "_")
    print(country)
    country_dir = main_dir+country
    if not os.path.exists(country_dir):
        os.mkdir(country_dir)
    files = [f for f in os.listdir(country_dir) if os.path.isfile(os.path.join(country_dir, f))]
    filename = f"{len(files)}.png"
    img.save(country_dir+"/"+filename)

def main():
    handle = Handle()
    with open("omni/temp_coords.txt", "r") as f:
        data = f.readlines()
    coords = []
    for line in data:
        coords.append(Coordinate(*map(float, line.split(", "))))
    batches = [coords[x:x+100] for x in range(0, len(coords), 100)]
    for batch in batches:
        images = handle.get_full_panos(batch, 0)
        for country, img in images:
            if img:
                save_img(country, img)

if __name__ == "__main__":
    #cProfile.run("main()", sort="cumtime")
    main()
