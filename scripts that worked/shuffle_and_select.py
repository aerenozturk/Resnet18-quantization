import os
import random
import shutil

# Define the source and destination folders
source_folder = "merged_dataset"
random_sampled_folder = "random_sampled_dataset"
remaining_folder = "remaining_dataset"

# Ensure the destination folders exist
os.makedirs(random_sampled_folder, exist_ok=True)
os.makedirs(remaining_folder, exist_ok=True)

# Get all image files from the source folder
all_images = sorted([f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Shuffle the images randomly
random.shuffle(all_images)

# Counters for renumbering
random_sample_counter = 1
remaining_counter = 1

# Process the images
for index, file in enumerate(all_images):
    src_path = os.path.join(source_folder, file)
    
    # Move every 20th image to the `random_sampled_dataset`
    if (index + 1) % 20 == 0:
        new_file_name = f"{random_sample_counter:03d}.jpg"
        dest_path = os.path.join(random_sampled_folder, new_file_name)
        shutil.move(src_path, dest_path)
        print(f"Moved to Random Sampled: {dest_path}")
        random_sample_counter += 1
    else:
        new_file_name = f"{remaining_counter:03d}.jpg"
        dest_path = os.path.join(remaining_folder, new_file_name)
        shutil.move(src_path, dest_path)
        print(f"Moved to Remaining Dataset: {dest_path}")
        remaining_counter += 1

print(f"\n✅ Process complete! Random sampled images: {random_sample_counter-1}, Remaining images: {remaining_counter-1}")

