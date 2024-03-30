import os
from zipfile import ZipFile
from PIL import Image

def resize_images_in_zip(zip_file_path, output_zip_path, target_size=(224, 224)):
    # Create a temporary directory to store resized images
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Extract images from the zip file
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Resize images
    for root, dirs, files in os.walk(temp_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                with Image.open(file_path) as img:
                    img_resized = img.resize(target_size)
                    img_resized.save(file_path)
            except IOError as e:
                print(e)
                print(f"Failed to resize {filename}")
    
    # Create a new zip file with resized images
    with ZipFile(output_zip_path, 'w') as new_zip:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                new_zip.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))
    
    # Clean up temporary directory
    for root, dirs, files in os.walk(temp_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(temp_dir)

# Example usage:
zip_file_path = "COVID-19_Radiography.zip"
output_zip_path = "resized_output.zip"
resize_images_in_zip(zip_file_path, output_zip_path)
