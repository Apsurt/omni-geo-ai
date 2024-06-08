"""
Docstring
"""

import os
import io
from typing import List, Dict, Tuple
import json
import re
import requests
from PIL import Image
from resources import Coordinate, combine_images
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
        n = len(coordinates)
        for idx, coord in enumerate(coordinates):
            per = round((idx+1)/n*100, 2)
            print(f"Getting pano_ids: {per}%", end="\r")
            url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={coord.lat},{coord.lng}&key={self.__key}"
            response = requests.post(url, headers=headers, timeout=10)
            response_json = json.loads(response.text)
            if response_json["status"] == "OK":
                pano_ids.append(response_json["pano_id"])
            else:
                pano_ids.append(None)
        print()
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
        try:
            components = metadata["addressComponents"]
        except KeyError as e:
            print(metadata)
            raise e
        for component in components:
            if "country" in component["types"]:
                return component

    def get_tile(self, pano_id: str, z: int, x: int, y: int) -> Image:
        headers = {"Content-Type": "application/json"}
        url = f"https://tile.googleapis.com/v1/streetview/tiles/{z}/{x}/{y}?session={self.__session_id}&key={self.__key}&panoId={pano_id}"
        image_bytes = requests.get(url, headers=headers, timeout=10)
        self.analyze_response(image_bytes)
        return Image.open(io.BytesIO(image_bytes.content))

    def get_full_panos(self, coordinates: Coordinate | List[Coordinate]) -> List[Tuple[str, Image]]:
        pano_ids = self.get_pano_ids(coordinates)
        n = len(pano_ids)
        z = 0
        combined_images = []
        img_got = 0 
        for idx, pano_id in enumerate(pano_ids):
            per = round((idx+1)/n*100, 2)
            print(f"Getting images: {per}%", end="\r")
            if pano_id is None:
                combined_images.append((None, None))
                continue
            img_got += 1
            metadata = self.get_metadata(pano_id)
            country_dict = self.get_country_from_metadata(metadata)
            country = country_dict["longName"]
            image = self.get_tile(pano_id, z, 0, 0)
            combined_image = combine_images(image)
            combined_images.append((country, combined_image))
        print()
        print(f"Got {img_got} images")
        return combined_images

def main():
    c = Handle()
    coord = Coordinate(18.35928297441725, -66.07041677031006)
    a = c.get_full_panos(coord)
    print(a)

if __name__ == "__main__":
    main()