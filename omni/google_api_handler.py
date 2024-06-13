"""Handle with google api."""

from __future__ import annotations

import io
import json
import os
import re

import requests
from dotenv import load_dotenv
from PIL import Image
from resources import Coordinate, combine_images

load_dotenv()

MAX_GOOGLE_API_LOCATIONS_COUNT = 100

class Handle:
    """Class handling Google API."""

    def __init__(self: Handle) -> None:
        """Handle constructor."""
        self.__session_id = os.getenv("GOOGLE_SESSION")
        self.__key = os.getenv("GOOGLE_KEY")
        self.update_session_id()

    def analyze_response(self: Handle, response: requests.Response) -> None:
        """Anlyze response and catch errors.

        :param self: self
        :type self: Handle
        :param response: response to check
        :type response: requests.Response
        :raises RuntimeError: code 400
        :raises RuntimeError: code 404
        """
        code = int(re.findall(r"\d+", str(response))[0])
        msg = "Bad request\n" + response.text
        match code:
            case 200:
                pass
            case 400:
                raise RuntimeError(msg)
            case 404:
                raise RuntimeError(msg)

    def get_session_id(self: Handle) -> str:
        """Get session id from google.

        :param self: self
        :type self: Handle
        :return: session id
        :rtype: str
        """
        headers = {"Content-Type": "application/json"}
        params = {"mapType":  "streetview",
                  "language": "en-US",
                  "region":   "US"}
        url = f"https://tile.googleapis.com/v1/createSession?key={self.__key}"
        response = requests.post(url, headers=headers, params=params, timeout=10)
        self.analyze_response(response)
        return json.loads(response.text)["session"]

    def update_session_id(self: Handle) -> None:
        """Update session id.

        Gets session id from google, then saves it in class and in .env file.

        :param self: self
        :type self: Handle
        """
        new_session_id = self.get_session_id()
        self.__session_id = new_session_id
        os.environ["GOOGLE_SESSION"] = new_session_id

    def get_pano_ids(
        self: Handle,
        coordinates: Coordinate | list[Coordinate],
        ) -> list[str]:
        """Get panorama ids from google.

        Gets up to 100 panorama ids from list of locations.

        :param self: self
        :type self: Handle
        :param coordinates: list of coordiantes
        :type coordinates: Coordinate | list[Coordinate]
        :raises RuntimeError: More than 100 locations
        :return: list of pano ids
        :rtype: list[str]
        """
        if isinstance(coordinates, Coordinate):
            coordinates = [coordinates]
        if len(coordinates) > MAX_GOOGLE_API_LOCATIONS_COUNT:
            msg = "Too much coordinates for one response."
            raise RuntimeError(msg)
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

    def get_metadata(self: Handle, pano_id: str) -> dict:
        """Get metadata from pano id.

        :param self: self
        :type self: Handle
        :param pano_id: pano id to get metadata for
        :type pano_id: str
        :return: metadata
        :rtype: dict
        """
        headers = {"Content-Type": "application/json"}
        params = {"mapType":  "streetview",
                  "language": "en-US",
                  "region":   "US"}
        url = f"https://tile.googleapis.com/v1/streetview/metadata?session={self.__session_id}&key={self.__key}&panoId={pano_id}"
        response = requests.get(url, headers=headers, params=params, timeout=10)
        self.analyze_response(response)
        return json.loads(response.text)

    def get_country_from_metadata(self: Handle, metadata: dict) -> dict:
        """Search metadata for country name.

        :param self: self
        :type self: Handle
        :param metadata: metadata
        :type metadata: dict
        :return: country component
        :rtype: dict
        """
        components = metadata["addressComponents"]
        for component in components:
            if "country" in component["types"]:
                return component
        return {}

    def get_tile(self: Handle, pano_id: str, z: int, x: int, y: int) -> Image:
        """Get street view tile.

        :param self: self
        :type self: Handle
        :param pano_id: pano id
        :type pano_id: str
        :param z: zoom level
        :type z: int
        :param x: x coordinate in panorama grid
        :type x: int
        :param y: y coordinate in panorama grid
        :type y: int
        :return: tile
        :rtype: Image
        """
        headers = {"Content-Type": "application/json"}
        url = f"https://tile.googleapis.com/v1/streetview/tiles/{z}/{x}/{y}?session={self.__session_id}&key={self.__key}&panoId={pano_id}"
        image_bytes = requests.get(url, headers=headers, timeout=10)
        self.analyze_response(image_bytes)
        return Image.open(io.BytesIO(image_bytes.content))

    def get_full_panos(
        self: Handle,
        coordinates: Coordinate | list[Coordinate],
        ) -> list[tuple[str, Image]]:
        """Get full panoramas from list of coordiantes.

        :param self: self
        :type self: Handle
        :param coordinates: list of coordinates to get panorama for
        :type coordinates: Coordinate | list[Coordinate]
        :return: list of images with country names
        :rtype: list[tuple[str, Image]]
        """
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
            metadata = self.get_metadata(pano_id)
            try:
                country_dict = self.get_country_from_metadata(metadata)
            except KeyError:
                continue
            try:
                country = country_dict["longName"]
            except TypeError:
                continue
            img_got += 1
            image = self.get_tile(pano_id, z, 0, 0)
            combined_image = combine_images(image)
            combined_images.append((country, combined_image))
        print()
        print(f"Got {img_got} images")
        return combined_images
