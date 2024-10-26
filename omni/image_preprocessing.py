import logging

from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    )
logger = logging.getLogger(__name__)

def process_image(img, file_path=None):
    try:
        # Find the first non-black row from the bottom
        width, height = img.size
        pixels = img.load()
        first_non_black = height - 1
        while first_non_black > 0 and all(pixels[x, first_non_black][0] == 0 for x in range(width)):
            first_non_black -= 1

        # Crop the image to remove black strip
        img_cropped = img.crop((0, 0, width, first_non_black + 1))

        # Resize back to 512x256
        img_resized = img_cropped.resize((512, 256), Image.LANCZOS)

        # Split into two 256x256 images
        left_half = img_resized.crop((0, 0, 256, 256))
        right_half = img_resized.crop((256, 0, 512, 256))

        if file_path:
            msg = f"Successfully processed {file_path}"
            logger.debug(msg)
        return left_half, right_half
    except Exception as e:
        if file_path:
            msg = f"Error processing {file_path}: {e}"
            logger.exception(msg)
        return None, None
