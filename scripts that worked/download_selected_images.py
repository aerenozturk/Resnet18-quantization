import json
from bing_image_downloader import downloader

# Load the selected categories JSON file
with open('selected_300_categories.json', 'r') as file:
    selected_categories = json.load(file)

# Create a list of category names (just the names)
category_names = list(selected_categories.values())

# Download images for each selected category
for category in category_names:
    print(f"Downloading images for: {category}")
    downloader.download(category, 
                        limit=13,  # Adjust the number of images per category as needed
                        output_dir=f'./test_dataset/{category}', 
                        adult_filter_off=True, 
                        force_replace=False, 
                        timeout=360, 
                        verbose=True)

print("✅ Image download complete!")

