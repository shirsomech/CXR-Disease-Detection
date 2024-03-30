import os
import cv2

def convert_to_rgb_and_save(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over all files in the input folder and its subfolders
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            input_path = os.path.join(root, file_name)
            
            # Check if the file is an image
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                try:
                    img = cv2.imread(input_path)
                    # Convert grayscale images to RGB
                    print(img.shape)
                    if len(img.shape) == 2:  # Grayscale image has shape (height, width)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        # Determine the relative path of the input file within the input folder
                        relative_path = os.path.relpath(input_path, input_folder)
                        # Construct the output path in the output folder
                        output_path = os.path.join(output_folder, relative_path)
                        # Replace the file extension with "_rgb.jpg"
                        output_path = os.path.splitext(output_path)[0] + "_rgb.jpg"
                        # Create the output folder for the current file if it doesn't exist
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        # Save the RGB image
                        cv2.imwrite(output_path, img_rgb)
                        print("Converted", input_path, "to RGB and saved as", output_path)
                    else:
                        print("Skipping", input_path, "- Image is already in RGB format.")
                except Exception as e:
                    print("Error processing", input_path, ":", e)

# Example usage
input_folder = "vinxray"
output_folder = "vinxrayrgb"
convert_to_rgb_and_save(input_folder, output_folder)
