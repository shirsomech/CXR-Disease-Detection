import os
import logging

def is_valid_image_file(filename):
  # Check file name extension
  valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
  if os.path.splitext(filename)[1].lower() not in valid_extensions:
    logging.info(f"Invalid image file extension \"{filename}\". Skipping this file...")
    return False

  return True
