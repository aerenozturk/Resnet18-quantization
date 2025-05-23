import os
import shutil
from PIL import Image

# Define the source folders and the destination folder
source_folders = ["dataset", "test_dataset"]
destination_folder = "merged_dataset"
target_size = (224, 224)  # Resize target size

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Counter for sequential numbering
image_counter = 1

# Loop through the source folders and collect all images
for folder in source_folders:
    for root, dirs, files in os.walk(folder):
        for file in files:
            # Only process image files (you can expand this list if needed)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Construct full file paths
                src_path = os.path.join(root, file)
                dest_path = os.path.join(destination_folder, f"{image_counter:04d}.jpg")
                
                # Try to open and resize the image
                try:
                    with Image.open(src_path) as img:
                        img_resized = img.resize(target_size)
                        img_resized.save(dest_path)
                        print(f"Processed: {dest_path}")
                        image_counter += 1
                except Exception as e:
                    print(f"Error processing {src_path}: {e}")

print(f"\n✅ Process complete! Total images processed: {image_counter-1}")

