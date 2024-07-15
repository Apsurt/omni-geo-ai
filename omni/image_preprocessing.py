import os
from PIL import Image
import json
import logging
from tqdm import tqdm
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
            logger.debug(f"Successfully processed {file_path}")
        return left_half, right_half
    except Exception as e:
        if file_path:
            logger.error(f"Error processing {file_path}: {str(e)}")
        return None, None

def process_directory(base_path, progress_file='progress.json'):
    # Load progress if exists
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        logger.info(f"Resuming from {progress['last_processed']}")
    else:
        progress = {'last_processed': None, 'processed_count': 0, 'error_count': 0}
    
    start_time = time.time()
    total_files = sum([len(files) for r, d, files in os.walk(base_path) if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files)])
    
    with tqdm(total=total_files, initial=progress['processed_count'], desc="Processing images") as pbar:
        # Traverse the directory structure
        for root, dirs, files in os.walk(base_path):
            # Skip directories we've already processed
            if progress['last_processed'] and root <= progress['last_processed']:
                continue
            
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for file in image_files:
                file_path = os.path.join(root, file)
                
                try:
                    # Process the image
                    with Image.open(file_path) as img:
                        if img.size == (256, 256):
                            pbar.update(1)
                            continue
                        left_half, right_half = process_image(img, file_path)
                    
                    if left_half is not None and right_half is not None:
                        # Save the processed images
                        base_name = os.path.splitext(file)[0]
                        left_half.save(os.path.join(root, f"{base_name}_left.png"))
                        right_half.save(os.path.join(root, f"{base_name}_right.png"))
                        
                        # Remove the original file
                        os.remove(file_path)
                        
                        progress['processed_count'] += 1
                    else:
                        progress['error_count'] += 1
                    
                    # Update and save progress
                    progress['last_processed'] = root
                    with open(progress_file, 'w') as f:
                        json.dump(progress, f)
                    
                    pbar.update(1)
                
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    progress['error_count'] += 1
            
            logger.info(f"Processed directory: {root}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"Processing complete. Total time: {total_time:.2f} seconds")
    logger.info(f"Total images processed: {progress['processed_count']}")
    logger.info(f"Total errors encountered: {progress['error_count']}")

if __name__ == "__main__":
    base_path = 'data/countries'
    process_directory(base_path)